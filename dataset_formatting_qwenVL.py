import json
import random
import os
from collections import defaultdict

# Load your validated dataset
with open("extracted_chart_specs_audited_noTrash.json", "r", encoding="utf-8") as f:
    raw_dataset = json.load(f)

SYSTEM_PROMPT = "Extract the Extended ChartSpec from this chart image. Return only the valid JSON."
stratification_pools = defaultdict(list)

# 1. Format and Group by Type/Sub-Type
for record in raw_dataset:
    spec = record["ChartSpec"]
    minified_spec = json.dumps(spec, ensure_ascii=False, separators=(',', ':'))

    qwen_format = {
        "id": record["id"],
        "images": [record["image"]],
        "messages": [
            {
                "role": "user",
                "content": f"<|image_pad|>\n{SYSTEM_PROMPT}"
            },
            {
                "role": "assistant",
                "content": minified_spec
            }
        ]
    }

    # Safely extract primary and secondary topology based on new schema
    c_type = spec.get("chart_type", "Unknown")

    # Handle single or multi-panel sub-types
    sub_type = "standard"
    if "panels" in spec and len(spec["panels"]) > 0:
        sub_type = spec["panels"][0].get("topo", {}).get("sub", "standard")

    pool_key = f"{c_type}_{sub_type}"
    stratification_pools[pool_key].append(qwen_format)

# 2. Stratified Splitting (80/10/10)
train_set, val_set, test_set = [], [], []

for key, items in stratification_pools.items():
    random.shuffle(items)
    n = len(items)

    # Safety Check: If a class has fewer than 3 samples, it can't be split across Train/Val/Test
    # Force it into Train so the model at least learns the syntax
    if n < 3:
        train_set.extend(items)
    else:
        train_split = int(n * 0.8)
        val_split = int(n * 0.9)

        train_set.extend(items[:train_split])
        val_set.extend(items[train_split:val_split])
        test_set.extend(items[val_split:])

# Shuffle the final sets
random.shuffle(train_set)
random.shuffle(val_set)
random.shuffle(test_set)

# 3. Save the Splits


def save_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


save_json(train_set, "qwen25_vl_3b_train.json")
save_json(val_set, "qwen25_vl_3b_val.json")
save_json(test_set, "qwen25_vl_3b_test.json")

print(f"✅ Split Complete:")
print(f"   - Train: {len(train_set)} samples")
print(f"   - Val:   {len(val_set)} samples")
print(f"   - Test:  {len(test_set)} samples")
