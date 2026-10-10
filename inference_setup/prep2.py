import os
import glob
import json
from collections import Counter, defaultdict
from datasets import Dataset

# --- Versioning Config ---
VERSION = "v2_10class" 
MAPPING_FILE = f"label_mappings_{VERSION}.json"

def map_l3_label(raw_label):
    if not raw_label: return None
    raw = raw_label.lower().replace('\n', ' ').strip()
    
    if any(k in raw for k in ["identity", "verified", "confirmed", "unconfirmed", "theft", "fraud", "address", "number change", "contact number"]): return "Identity & Verification"
    if any(k in raw for k in ["salary", "funds", "promise to pay", "payment commitment", "ptp"]): return "Commitment & Salary"
    if any(k in raw for k in ["conditional", "broken", "repeated", "avoidance", "withdrawn", "vague"]): return "Broken & Vague Commitments"
    if any(k in raw for k in ["settlement", "counter-offer", "negotiation", "written offer"]): return "Settlement & Negotiation"
    if any(k in raw for k in ["hardship", "vulnerability", "medical", "job loss", "income disruption", "emergency", "distress", "deceased", "overindebtedness", "multiple loans", "other lender"]): return "Hardship & Vulnerability"
    if any(k in raw for k in ["interest", "penalty", "penalties", "fee", "waiver", "outstanding amount", "disputes amount"]): return "Dispute - Financial"
    if any(k in raw for k in ["bureau", "already made", "already paid", "closed", "noc issued", "legitimacy", "fraudulently taken"]): return "Dispute - Bureau & Claims"
    if any(k in raw for k in ["frequency", "harassment", "rude", "stop calls", "email-only", "dnc", "refusal", "not interested", "hours"]): return "Complaints & DNC"
    if any(k in raw for k in ["legal", "lawyer", "rbi", "forum", "complaint", "threat", "insolvency", "court"]): return "Legal & Regulatory"
    if any(k in raw for k in ["callback", "follow-up", "agent action", "information pending", "third party", "co-applicant", "guarantor", "gateway", "upi failure", "hold"]): return "Follow-Up & Operations"
    return None

def load_disposenet_dataset(data_dir="datasets", mapping_file=MAPPING_FILE):
    print(f"Scanning '{data_dir}' for JSON files...")
    files = glob.glob(os.path.join(data_dir, '**', '*.json'), recursive=True)
    raw_data = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as fh:
            try:
                data = json.load(fh)
                if isinstance(data, list):
                    raw_data.extend(data)
                elif isinstance(data, dict):
                    raw_data.append(data)
            except json.JSONDecodeError:
                continue

    unique_l1, unique_l2 = set(), set()
    master_l3 = sorted([
        "Identity & Verification", "Commitment & Salary", "Broken & Vague Commitments",
        "Settlement & Negotiation", "Hardship & Vulnerability", "Dispute - Financial",
        "Dispute - Bureau & Claims", "Complaints & DNC", "Legal & Regulatory", "Follow-Up & Operations"
    ])
    
    for item in raw_data:
        l1 = item.get("L1", "").strip()
        l2 = item.get("L2", "").strip()
        if l1: unique_l1.add(l1)
        if l2: unique_l2.add(l2)

    unique_l1, unique_l2 = sorted(list(unique_l1)), sorted(list(unique_l2))
    l1_to_id = {l: i for i, l in enumerate(unique_l1)}
    l2_to_id = {l: i for i, l in enumerate(unique_l2)}
    l3_to_id = {l: i for i, l in enumerate(master_l3)}

    transcripts, labels_l1, labels_l2, labels_l3 = [], [], [], []
    l1_counts = Counter()
    l1_to_valid_l2 = defaultdict(set)

    for item in raw_data:
        t = item.get("transcript", "").strip()
        l1_s = item.get("L1", "").strip()
        l2_s = item.get("L2", "").strip()
        l3_r = item.get("L3", "")

        if not t or not l1_s or not l2_s: 
            continue

        l1_id, l2_id = l1_to_id[l1_s], l2_to_id[l2_s]
        transcripts.append(t)
        labels_l1.append(l1_id)
        labels_l2.append(l2_id)
        
        l1_counts[l1_id] += 1
        l1_to_valid_l2[l1_id].add(l2_id)
        
        l3_v = [0.0] * 10
        if l3_r:
            for f in [f.strip() for f in l3_r.split('|') if f.strip()]:
                m = map_l3_label(f)
                if m: 
                    l3_v[l3_to_id[m]] = 1.0
        labels_l3.append(l3_v)

    total_samples = len(labels_l1)
    num_l1 = len(unique_l1)
    l1_weights = [1.0] * num_l1
    for i in range(num_l1):
        if l1_counts[i] > 0:
            l1_weights[i] = total_samples / (num_l1 * l1_counts[i])

    mappings = {
        "l1": unique_l1, "l2": unique_l2, "l3": master_l3,
        "l1_to_id": l1_to_id, "l2_to_id": l2_to_id, "l3_to_id": l3_to_id,
        "l1_weights": l1_weights,
        "l1_to_valid_l2": {str(k): list(v) for k, v in l1_to_valid_l2.items()}
    }

    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mappings, f, indent=2)
    
    ds = Dataset.from_dict({"transcript": transcripts, "labels_l1": labels_l1, "labels_l2": labels_l2, "labels_l3": labels_l3})
    print(f"Non-destructive: Saved new mappings to {mapping_file}")
    return ds, mappings

if __name__ == "__main__":
    ds, mappings = load_disposenet_dataset()
    print(f"L1: {len(mappings['l1'])} | L2: {len(mappings['l2'])} | L3: {len(mappings['l3'])}")
