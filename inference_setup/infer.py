import os
import torch
import json
import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer
from safetensors.torch import load_model
from t2 import DisposeNet

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
MODEL_BASE     = "ai4bharat/indic-bert"
CHECKPOINT_DIR = "./outputs_v2_10class_base/checkpoint-945"
MAPPING_FILE   = "label_mappings_v2_10class.json"
INPUT_JSON     = "inference_setup/test_cases_clean_v2_10class_base.json"
OUTPUT_JSON    = "infer_results.json"
VIZ_DIR        = "explanations"          # folder for HTML visualizations

EXPLAIN        = False   # True  → IG attribution + HTML viz per sample (slow, max 10 samples)
                         # False → pure inference, no gradients (fast, all samples)
IG_STEPS       = 20      # only used when EXPLAIN=True (20=fast, 50=accurate)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. LOAD MODEL & MAPPINGS ──────────────────────────────────────────────────
with open(MAPPING_FILE, 'r') as f:
    mappings = json.load(f)

l1_names = mappings["l1"]
l2_names = mappings["l2"]
l3_names = mappings["l3"]
l1_to_id = mappings["l1_to_id"]
l2_to_id = mappings["l2_to_id"]
l3_to_id = mappings["l3_to_id"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)

model = DisposeNet(MODEL_BASE, mappings, len(l1_names), len(l2_names), len(l3_names))
load_model(model, f"{CHECKPOINT_DIR}/model.safetensors")
model.to(device)
model.eval()

embedding_layer = model.encoder.get_input_embeddings()

if EXPLAIN:
    os.makedirs(VIZ_DIR, exist_ok=True)


# ── 2. CHUNKED TOKENIZATION (mirrors training exactly) ───────────────────────
def chunked_tokenize(text):
    raw_ids = tokenizer(text, add_special_tokens=False)['input_ids']

    all_ids, all_masks, chunk_token_spans = [], [], []
    cursor = 0

    for _ in range(4):
        if cursor >= len(raw_ids):
            all_ids.append([tokenizer.pad_token_id] * 512)
            all_masks.append([0] * 512)
            chunk_token_spans.append((cursor, cursor))
            continue

        chunk_raw = raw_ids[cursor: cursor + 510]
        end       = cursor + len(chunk_raw)
        chunk_token_spans.append((cursor, end))
        cursor    = end

        new_ids = [tokenizer.cls_token_id] + chunk_raw + [tokenizer.sep_token_id]
        pad     = 512 - len(new_ids)
        all_ids.append(new_ids + [tokenizer.pad_token_id] * pad)
        all_masks.append([1] * len(new_ids) + [0] * pad)

    input_ids      = torch.tensor([all_ids],   dtype=torch.long, device=device)
    attention_mask = torch.tensor([all_masks], dtype=torch.long, device=device)
    return input_ids, attention_mask, raw_ids, chunk_token_spans


# ── 3. FAST INFERENCE (no grad) ───────────────────────────────────────────────
@torch.no_grad()
def infer(text):
    input_ids, attention_mask, _, _ = chunked_tokenize(text)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return (
        outputs["logits_l1"][0].cpu().numpy(),
        outputs["logits_l2"][0].cpu().numpy(),
        outputs["logits_l3"][0].cpu().numpy(),
    )


# ── 4. EXPLAIN PATH: Integrated Gradients ────────────────────────────────────
def _forward_with_embeddings(flat_embeddings, attention_mask, target_head, target_class_idx):
    batch_size, num_chunks, seq_len = 1, 4, 512
    flat_attn   = attention_mask.view(-1, seq_len)
    is_empty    = (flat_attn.sum(dim=-1) == 0)
    safe_attn   = flat_attn.clone()
    safe_attn[is_empty, 0] = 1

    outputs     = model.encoder(inputs_embeds=flat_embeddings, attention_mask=safe_attn)
    cls_tokens  = outputs.last_hidden_state[:, 0, :].view(batch_size, num_chunks, -1)
    is_empty_3d = is_empty.view(batch_size, num_chunks, 1)
    cls_tokens  = cls_tokens.masked_fill(is_empty_3d, -10000.0)

    pooled, _   = torch.max(cls_tokens, dim=1)
    pooled      = model.dropout(pooled).to(model.l1_head.weight.dtype)

    head_map = {"l1": model.l1_head, "l2": model.l2_head, "l3": model.l3_head}
    return head_map[target_head](pooled)[0, target_class_idx]


def _get_raw_ids_and_spans(input_ids):
    raw_ids, spans, cursor = [], [], 0
    for chunk in input_ids[0].tolist():
        chunk_raw = []
        for tok in chunk[1:]:
            if tok in (tokenizer.sep_token_id, tokenizer.pad_token_id):
                break
            chunk_raw.append(tok)
        spans.append((cursor, cursor + len(chunk_raw)))
        raw_ids.extend(chunk_raw)
        cursor += len(chunk_raw)
    return raw_ids, spans


def _integrated_gradients(input_ids, attention_mask, target_head, target_class_idx):
    raw_ids, chunk_token_spans = _get_raw_ids_and_spans(input_ids)

    flat_ids  = input_ids.view(-1, 512)
    with torch.no_grad():
        real_embs = embedding_layer(flat_ids).float()

    baseline          = torch.zeros_like(real_embs)
    accumulated_grads = torch.zeros_like(real_embs)

    for step in range(IG_STEPS):
        alpha        = step / max(IG_STEPS - 1, 1)
        interp       = (baseline + alpha * (real_embs - baseline)).detach().requires_grad_(True)
        target_logit = _forward_with_embeddings(interp, attention_mask, target_head, target_class_idx)
        model.zero_grad()
        target_logit.backward()
        accumulated_grads += interp.grad.detach()

    avg_grads = accumulated_grads / IG_STEPS
    ig_attrs  = ((real_embs - baseline) * avg_grads).sum(dim=-1)  # [4, 512]

    ig_scores = []
    for chunk_idx, (start, end) in enumerate(chunk_token_spans):
        chunk_len = end - start
        if chunk_len == 0:
            continue
        ig_scores.extend(ig_attrs[chunk_idx, 1: 1 + chunk_len].cpu().numpy().tolist())

    return np.array(ig_scores), raw_ids


def _aggregate_subwords(raw_ids, ig_scores):
    """Merge SentencePiece subwords → words, sum their IG scores, normalize to [-1, 1]."""
    tokens = tokenizer.convert_ids_to_tokens(raw_ids)
    words, word_scores = [], []
    for token, score in zip(tokens, ig_scores):
        clean = token.replace('▁', '')
        if token.startswith('▁') or not words:
            words.append(clean)
            word_scores.append(float(score))
        else:
            words[-1]      += clean
            word_scores[-1] += float(score)

    scores_arr = np.array(word_scores)
    max_abs    = np.abs(scores_arr).max()
    if max_abs > 1e-8:
        scores_arr = scores_arr / max_abs
    return list(zip(words, scores_arr.tolist()))


def _top_contributors(word_score_pairs, top_pct=0.10):
    ranked = sorted(word_score_pairs, key=lambda x: x[1], reverse=True)
    n = max(1, int(len(ranked) * top_pct))
    return {
        "pos_contributors": [(w, round(s, 4)) for w, s in ranked[:n]],
        "neg_contributors": [(w, round(s, 4)) for w, s in ranked[-n:]],
    }


def get_word_importance(text, target_head, target_class_idx):
    """Returns (top_contributors_dict, word_score_pairs) — pairs needed for viz."""
    input_ids, attention_mask, _, _ = chunked_tokenize(text)
    ig_scores, raw_ids = _integrated_gradients(input_ids, attention_mask, target_head, target_class_idx)
    word_scores = _aggregate_subwords(raw_ids, ig_scores)
    return _top_contributors(word_scores), word_scores


# ── 5. HTML VISUALIZATION ─────────────────────────────────────────────────────
def _score_to_color(score):
    """
    score in [-1, 1]
    Positive → green  (more intense = stronger positive signal)
    Negative → red    (more intense = stronger suppression)
    Near zero → white
    """
    alpha = abs(score)  # 0..1 intensity
    if score > 0:
        r = int(255 * (1 - alpha * 0.85))
        g = 255
        b = int(255 * (1 - alpha * 0.85))
    else:
        r = 255
        g = int(255 * (1 - alpha * 0.85))
        b = int(255 * (1 - alpha * 0.85))
    return f"rgb({r},{g},{b})"


def _render_highlighted_text(word_scores):
    """Return an HTML <span> string with each word colored by its IG score."""
    spans = []
    for word, score in word_scores:
        color   = _score_to_color(score)
        tooltip = f"score: {score:+.3f}"
        spans.append(
            f'<span style="background:{color};border-radius:3px;padding:1px 3px;margin:1px;'
            f'display:inline-block;font-size:14px;" title="{tooltip}">{word}</span>'
        )
    return " ".join(spans)


def save_viz_html(sample_idx, text, pred_l2, pred_l3_names,
                  l2_word_scores, l3_explanations, ground_truth=None):
    """
    Saves one HTML file per sample with:
      - Transcript info + predictions
      - L2 class: full word-highlight heatmap
      - Each active L3 flag: full word-highlight heatmap
    """
    gt_html = ""
    if ground_truth:
        correct_l2 = "✅" if ground_truth["L2"] == pred_l2 else "❌"
        gt_html = f"""
        <div class="gt-box">
            <b>Ground Truth</b> — L2: <code>{ground_truth['L2']}</code> {correct_l2}
            &nbsp;|&nbsp; L3: <code>{', '.join(ground_truth['L3']) or 'none'}</code>
        </div>"""

    # L2 block
    l2_html = f"""
    <div class="class-block">
        <div class="class-title">L2 Disposition: <span class="tag">{pred_l2}</span></div>
        <div class="legend">
            <span style="background:rgb(60,255,60);padding:2px 8px;border-radius:3px;">Positive (supports class)</span>
            &nbsp;
            <span style="background:rgb(255,60,60);padding:2px 8px;border-radius:3px;">Negative (suppresses class)</span>
        </div>
        <div class="text-block">{_render_highlighted_text(l2_word_scores)}</div>
    </div>"""

    # L3 blocks (one per active flag)
    l3_html = ""
    for flag_name, word_scores in l3_explanations.items():
        l3_html += f"""
    <div class="class-block">
        <div class="class-title">L3 Flag: <span class="tag tag-l3">{flag_name}</span></div>
        <div class="text-block">{_render_highlighted_text(word_scores)}</div>
    </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Sample {sample_idx} Explanation</title>
<style>
  body      {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background: #f9f9f9; color: #222; }}
  h2        {{ border-bottom: 2px solid #333; padding-bottom: 6px; }}
  .meta     {{ background: #fff; border: 1px solid #ddd; border-radius:6px; padding:12px 16px; margin-bottom:16px; }}
  .gt-box   {{ background: #fffbe6; border: 1px solid #e6c800; border-radius:6px; padding:10px 14px; margin-top:8px; }}
  .class-block {{ background:#fff; border:1px solid #ddd; border-radius:6px; padding:14px 18px; margin-bottom:20px; }}
  .class-title {{ font-size:16px; font-weight:bold; margin-bottom:8px; }}
  .tag      {{ background:#2563eb; color:#fff; border-radius:4px; padding:2px 8px; font-size:13px; }}
  .tag-l3   {{ background:#7c3aed; }}
  .legend   {{ font-size:12px; margin-bottom:10px; }}
  .text-block {{ line-height: 2.2; word-spacing: 4px; }}
  .transcript {{ font-style:italic; color:#444; font-size:13px; margin-top:6px; white-space:pre-wrap; }}
</style>
</head>
<body>
<h2>Sample {sample_idx} — Word Attribution</h2>
<div class="meta">
  <b>Predictions</b> — L2: <code>{pred_l2}</code>
  &nbsp;|&nbsp; L3: <code>{', '.join(pred_l3_names) or 'none'}</code>
  {gt_html}
  <div class="transcript">{text[:500]}{'...' if len(text) > 500 else ''}</div>
</div>
{l2_html}
{l3_html}
</body>
</html>"""

    path = os.path.join(VIZ_DIR, f"sample_{sample_idx:03d}.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


# ── 6. LABEL DETECTION ────────────────────────────────────────────────────────
def has_labels(cases):
    for case in cases:
        return all(k in case for k in ("L1", "L2", "L3"))
    return False


def encode_l3(l3_raw):
    vec = [0.0] * len(l3_names)
    if l3_raw:
        for tag in [t.strip() for t in l3_raw.split('|') if t.strip()]:
            if tag in l3_to_id:
                vec[l3_to_id[tag]] = 1.0
    return vec


# ── 7. MAIN LOOP ──────────────────────────────────────────────────────────────
with open(INPUT_JSON, 'r') as f:
    test_cases = json.load(f)

labeled = has_labels(test_cases)
print(f"Labeled dataset : {labeled}")
print(f"Explain mode    : {EXPLAIN}")
print(f"Samples         : {len(test_cases)}\n")

if EXPLAIN:
    test_cases = test_cases[:10]

results = []
all_l1_true, all_l1_pred = [], []
all_l2_true, all_l2_pred = [], []
all_l3_true, all_l3_pred = [], []

for i, case in enumerate(test_cases):
    text = case['transcript'].strip()
    print(f"[{i+1}/{len(test_cases)}] {text[:60]}...")

    logits_l1, logits_l2, logits_l3 = infer(text)

    pred_l1_id  = int(np.argmax(logits_l1))
    pred_l2_id  = int(np.argmax(logits_l2))
    pred_l3_bin = (expit(logits_l3) > 0.5).astype(int).tolist()

    pred_l1      = l1_names[pred_l1_id]
    pred_l2      = l2_names[pred_l2_id]
    pred_l3      = [l3_names[j] for j, v in enumerate(pred_l3_bin) if v == 1]

    entry = {
        "transcript_preview": text[:100],
        "predictions": {"L1": pred_l1, "L2": pred_l2, "L3": pred_l3},
    }

    # ── Ground truth ─────────────────────────────────────────────────────────
    ground_truth_entry = None
    if labeled:
        true_l1   = case['L1'].strip()
        true_l2   = case['L2'].strip()
        true_l3_v = encode_l3(case.get('L3', ''))
        true_l3   = [l3_names[j] for j, v in enumerate(true_l3_v) if v == 1.0]

        true_l1_id = l1_to_id.get(true_l1, -1)
        true_l2_id = l2_to_id.get(true_l2, -1)

        entry["ground_truth"] = {"L1": true_l1, "L2": true_l2, "L3": true_l3}
        entry["correct"]      = {
            "L1": pred_l1 == true_l1,
            "L2": pred_l2 == true_l2,
            "L3": pred_l3_bin == [int(v) for v in true_l3_v],
        }
        ground_truth_entry = {"L2": true_l2, "L3": true_l3}

        if true_l1_id != -1:
            all_l1_true.append(true_l1_id); all_l1_pred.append(pred_l1_id)
        if true_l2_id != -1:
            all_l2_true.append(true_l2_id); all_l2_pred.append(pred_l2_id)
        all_l3_true.append([int(v) for v in true_l3_v])
        all_l3_pred.append(pred_l3_bin)

    # ── Explanation + visualization ───────────────────────────────────────────
    if EXPLAIN:
        # L2
        l2_top, l2_word_scores = get_word_importance(text, "l2", pred_l2_id)

        # All active L3 flags (not just the first one)
        l3_explanations_top  = {}   # for JSON
        l3_explanations_viz  = {}   # word_scores per flag for HTML
        for j, v in enumerate(pred_l3_bin):
            if v == 1:
                flag_name = l3_names[j]
                top, word_scores = get_word_importance(text, "l3", j)
                l3_explanations_top[flag_name] = top
                l3_explanations_viz[flag_name] = word_scores

        entry["explanations"] = {"L2": l2_top, "L3": l3_explanations_top}

        # Save HTML visualization
        viz_path = save_viz_html(
            sample_idx      = i + 1,
            text            = text,
            pred_l2         = pred_l2,
            pred_l3_names   = pred_l3,
            l2_word_scores  = l2_word_scores,
            l3_explanations = l3_explanations_viz,
            ground_truth    = ground_truth_entry,
        )
        entry["viz_path"] = viz_path
        print(f"         → viz saved: {viz_path}")

    results.append(entry)

# ── 8. F1 REPORTING (labeled only) ───────────────────────────────────────────
summary = {}

if labeled:
    f1_l1 = f1_score(all_l1_true, all_l1_pred, average='macro', zero_division=0)
    f1_l2 = f1_score(all_l2_true, all_l2_pred, average='macro', zero_division=0)
    f1_l3 = f1_score(all_l3_true, all_l3_pred, average='macro', zero_division=0)

    print("\n" + "=" * 55)
    print(f"  F1 L1 (Contact Type)   : {f1_l1:.4f}")
    print(f"  F1 L2 (Disposition)    : {f1_l2:.4f}")
    print(f"  F1 L3 (Flags, macro)   : {f1_l3:.4f}")
    print("=" * 55)

    print("\n── L2 Per-Class Report ──")
    print(classification_report(all_l2_true, all_l2_pred, labels=list(range(len(l2_names))), target_names=l2_names, zero_division=0))

    print("\n── L3 Per-Class Report ──")
    print(classification_report(all_l3_true, all_l3_pred, labels=list(range(len(l3_names))), target_names=l3_names, zero_division=0))

    summary = {
        "F1_L1_macro": round(f1_l1, 4),
        "F1_L2_macro": round(f1_l2, 4),
        "F1_L3_macro": round(f1_l3, 4),
    }

# ── 9. SAVE OUTPUT ────────────────────────────────────────────────────────────
output = {"summary": summary, "results": results}
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=4)

print(f"\nResults saved to {OUTPUT_JSON}")
if EXPLAIN:
    print(f"HTML visualizations saved to ./{VIZ_DIR}/")
