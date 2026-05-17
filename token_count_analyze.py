import json
import numpy as np
import os
from transformers import AutoTokenizer

# Initialize the exact tokenizer you are using for the 3B model
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
print(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Define the paths to your freshly split datasets
splits = {
    "Train": "qwen25_vl_3b_train.json",
    "Validation": "qwen25_vl_3b_val.json",
    "Test": "qwen25_vl_3b_test.json"
}


def analyze_token_counts(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    token_counts = []

    for item in data:
        # Extract the target output (the assistant's response)
        for msg in item["messages"]:
            if msg["role"] == "assistant":
                target_text = msg["content"]
                # Encode the text to get the exact token count
                tokens = tokenizer.encode(
                    target_text, add_special_tokens=False)
                token_counts.append(len(tokens))
                break  # Move to next item once assistant response is found

    return token_counts


# Analyze and print statistics for each split
overall_counts = []

for split_name, file_path in splits.items():
    counts = analyze_token_counts(file_path)
    if not counts:
        continue

    overall_counts.extend(counts)

    counts_array = np.array(counts)
    print(f"\n--- {split_name} Split ({len(counts)} samples) ---")
    print(f"Min tokens:    {np.min(counts_array)}")
    print(f"Max tokens:    {np.max(counts_array)}")
    print(f"Mean tokens:   {np.mean(counts_array):.1f}")
    print(f"Median tokens: {np.median(counts_array):.1f}")
    print(f"90th pctl:     {np.percentile(counts_array, 90):.1f}")
    print(f"95th pctl:     {np.percentile(counts_array, 95):.1f}")
    print(f"99th pctl:     {np.percentile(counts_array, 99):.1f}")

# Overall Dataset check
if overall_counts:
    overall_array = np.array(overall_counts)
    print(f"\n================ OVERALL DATASET ================")
    print(f"Absolute Max Tokens: {np.max(overall_array)}")

    # Check against typical generation limits
    # While Qwen2.5 can handle long contexts, output generation > 2048 in SFT can be memory heavy
    WARNING_THRESHOLD = 2048
    over_limit = np.sum(overall_array > WARNING_THRESHOLD)

    if over_limit > 0:
        print(
            f"⚠️ WARNING: {over_limit} samples have > {WARNING_THRESHOLD} output tokens.")
        print("Make sure your SFT `max_length` parameter in TrainingArguments can accommodate this!")
    else:
        print(
            f"✅ All samples fit comfortably within safe generation limits (< {WARNING_THRESHOLD} tokens).")
