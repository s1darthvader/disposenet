import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import f1_score
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EvalPrediction
)

# NOTE: Ensure your prep.py has the 10-class logic and outputs this specific mapping file
from prep2 import load_disposenet_dataset


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
VERSION = "v2_10class_base"
MAPPING_FILE = "label_mappings_v2_10class.json"
OUTPUT_DIR = f"./outputs_{VERSION}"
TEST_CSV_PATH = f"test_set_{VERSION}.csv"


# ── 1. Loss Logging Callback ──────────────────────────────────────────────────

class LossLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # DDP FIX: Only print on the main process to avoid log duplication
        if state.is_world_process_zero:
            loss = logs.get("loss")
            lr   = logs.get("learning_rate")
            if loss is not None:
                lr_str = f"  lr={lr:.2e}" if lr is not None else ""
                print(f"[Step {state.global_step:>5}]  loss={loss:.4f}{lr_str}")


# ── 2. Model ──────────────────────────────────────────────────────────────────

class DisposeNet(nn.Module):
    def __init__(self, encoder_path, mappings, num_l1=2, num_l2=45, num_l3=10):
        super().__init__()
        self.encoder  = AutoModel.from_pretrained(encoder_path)

        # --- DDP FIX: Freeze the unused pooler so PyTorch DDP ignores it safely ---
        if hasattr(self.encoder, 'pooler') and self.encoder.pooler is not None:
            for param in self.encoder.pooler.parameters():
                param.requires_grad = False

        hidden_size   = self.encoder.config.hidden_size
        self.dropout  = nn.Dropout(0.1)

        self.l1_head  = nn.Linear(hidden_size, num_l1)
        self.l2_head  = nn.Linear(hidden_size, num_l2)
        self.l3_head  = nn.Linear(hidden_size, num_l3)

        for head in [self.l1_head, self.l2_head, self.l3_head]:
            nn.init.normal_(head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(head.bias)

        self.register_buffer("valid_l2_for_l1_0", torch.tensor(mappings["l1_to_valid_l2"]["0"], dtype=torch.long))
        self.register_buffer("valid_l2_for_l1_1", torch.tensor(mappings["l1_to_valid_l2"]["1"], dtype=torch.long))
        self.register_buffer("l1_weights", torch.ones(num_l1))

    def set_l1_weights(self, weights: list, device):
        self.l1_weights = torch.tensor(weights, dtype=torch.float).to(device)

    def make_contiguous(self):
        for param in self.parameters():
            param.data = param.data.contiguous()

    def get_optimizer_groups(self, encoder_lr=2e-5, head_lr=1e-3):
        encoder_params = list(self.encoder.parameters())
        head_params    = (list(self.l1_head.parameters()) +
                          list(self.l2_head.parameters()) +
                          list(self.l3_head.parameters()))
        return [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params,    "lr": head_lr},
        ]

    def forward(self, input_ids, attention_mask=None, labels_l1=None, labels_l2=None, labels_l3=None):
        batch_size, num_chunks, seq_len = input_ids.size()

        flat_input_ids      = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)

        is_empty_chunk      = (flat_attention_mask.sum(dim=-1) == 0)
        safe_attention_mask = flat_attention_mask.clone()
        safe_attention_mask[is_empty_chunk, 0] = 1

        outputs = self.encoder(input_ids=flat_input_ids, attention_mask=safe_attention_mask)
        cls_tokens = outputs.last_hidden_state[:, 0, :].view(batch_size, num_chunks, -1)

        is_empty_chunk_3d = is_empty_chunk.view(batch_size, num_chunks, 1)
        cls_tokens        = cls_tokens.masked_fill(is_empty_chunk_3d, -10000.0)

        pooled_output, _  = torch.max(cls_tokens, dim=1)
        pooled_output     = self.dropout(pooled_output).to(self.l1_head.weight.dtype)

        logits_l1     = self.l1_head(pooled_output)
        raw_logits_l2 = self.l2_head(pooled_output)
        logits_l3     = self.l3_head(pooled_output)

        loss = None
        if labels_l1 is not None:
            # Weighted Loss for 10-Class merge: emphasizes specific flag extraction
            loss = (
                0.1 * nn.CrossEntropyLoss(weight=self.l1_weights.to(logits_l1.device))(logits_l1, labels_l1.long()) +
                0.4 * nn.CrossEntropyLoss()(raw_logits_l2, labels_l2.long()) +
                0.5 * nn.BCEWithLogitsLoss()(logits_l3, labels_l3.float())
            )

        l1_preds = torch.argmax(logits_l1, dim=-1)
        mask = torch.full_like(raw_logits_l2, -10000.0)
        for i in range(raw_logits_l2.size(0)):
            if l1_preds[i] == 0:
                mask[i, self.valid_l2_for_l1_0] = 0.0
            else:
                mask[i, self.valid_l2_for_l1_1] = 0.0

        return {
            "loss":      loss,
            "logits_l1": logits_l1,
            "logits_l2": raw_logits_l2 + mask,
            "logits_l3": logits_l3
        }


# ── 3. Trainer ────────────────────────────────────────────────────────────────

class DisposeTrainer(Trainer):
    def create_optimizer(self):
        param_groups   = self.model.get_optimizer_groups(encoder_lr=self.args.learning_rate, head_lr=1e-3)
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        l1 = inputs.pop("labels_l1")
        l2 = inputs.pop("labels_l2")
        l3 = inputs.pop("labels_l3")
        outputs = model(**inputs, labels_l1=l1, labels_l2=l2, labels_l3=l3)
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            l1 = inputs.pop("labels_l1")
            l2 = inputs.pop("labels_l2")
            l3 = inputs.pop("labels_l3")
            outputs = model(**inputs, labels_l1=l1, labels_l2=l2, labels_l3=l3)

        if prediction_loss_only:
            return (outputs["loss"], None, None)

        return (
            outputs["loss"],
            (outputs["logits_l1"], outputs["logits_l2"], outputs["logits_l3"]),
            (l1, l2, l3)
        )


# ── 4. Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    l1_p = np.argmax(logits[0], axis=-1)
    l2_p = np.argmax(logits[1], axis=-1)
    l3_p = (expit(logits[2]) > 0.5).astype(int)

    return {
        "F1_L1_Contact":     f1_score(labels[0], l1_p, average='macro', zero_division=0),
        "F1_L2_Disposition": f1_score(labels[1], l2_p, average='macro', zero_division=0),
        "F1_L3_Flags_Macro": f1_score(labels[2], l3_p, average='macro', zero_division=0),
    }


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_PATH = "ai4bharat/indic-bert"

    # Load data using the new mapping file
    ds, mappings = load_disposenet_dataset("datasets", mapping_file=MAPPING_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    def chunked_tokenize(examples):
        all_ids, all_masks = [], []
        tokenized = tokenizer(examples['transcript'], add_special_tokens=False)

        for ids in tokenized['input_ids']:
            c_ids, c_masks = [], []
            for i in range(0, len(ids), 384):
                if len(c_ids) == 4:
                    break
                chunk   = ids[i : i + 510]
                new_ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
                pad     = 512 - len(new_ids)

                c_ids.append(new_ids + [tokenizer.pad_token_id] * pad)
                c_masks.append([1] * len(new_ids) + [0] * pad)

            while len(c_ids) < 4:
                c_ids.append([tokenizer.pad_token_id] * 512)
                c_masks.append([0] * 512)

            all_ids.append(c_ids)
            all_masks.append(c_masks)

        return {"input_ids": all_ids, "attention_mask": all_masks}

    tokenized_ds = ds.map(chunked_tokenize, batched=True)
    tokenized_ds.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels_l1', 'labels_l2', 'labels_l3']
    )

    split = tokenized_ds.train_test_split(test_size=0.1, seed=42)

    num_l1 = len(mappings["l1"])
    num_l2 = len(mappings["l2"])
    num_l3 = len(mappings["l3"])

    if local_rank in [-1, 0]:
        print(f"Model heads — L1: {num_l1}  L2: {num_l2}  L3: {num_l3}")

        # Save test set for later offline inference
        test_df = pd.DataFrame(split["test"])
        test_df.to_csv(TEST_CSV_PATH, index=False)
        print(f"Test set exported to {TEST_CSV_PATH}")

    model = DisposeNet(MODEL_PATH, mappings, num_l1, num_l2, num_l3)
    model.set_l1_weights(mappings["l1_weights"], device=device)
    model.make_contiguous()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=50,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,  # Restored to original stable LR
        weight_decay=1e-2,
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="F1_L2_Disposition",
        greater_is_better=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        logging_steps=10,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = DisposeTrainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        compute_metrics=compute_metrics,
        callbacks=[LossLoggerCallback()],
    )

    trainer.train()

    if trainer.is_world_process_zero():
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        torch.save(trainer.model.state_dict(), os.path.join(OUTPUT_DIR, "disposenet_final.pt"))
        print(f"\nTraining complete. Model and Test CSV saved in version: {VERSION}")


if __name__ == "__main__":
    main()
