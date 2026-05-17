import json
import re
import os
import torch
import PIL.Image
from datasets import Dataset
from tqdm import tqdm
from unsloth import FastVisionModel

# ==========================================
# 1. CONFIGURATION
# ==========================================
WORKSPACE_DIR = "./"
LOCAL_IMAGE_DIR = os.path.join(WORKSPACE_DIR, "final_balanced_images")
TEST_DATASET_PATH = os.path.join(
    WORKSPACE_DIR, "qwen25_vl_3b_test.json")  # Update this
LORA_ADAPTER_PATH = "./qwen3b_lora_sft_bf16_final"
BASE_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# ==========================================
# 2. HELPER & REWARD FUNCTIONS
# ==========================================


def extract_json(text):
    text = text.strip()
    json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def robust_float(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            clean_val = val.strip().replace(',', '')
            return float(clean_val)
        except ValueError:
            raise ValueError(f"Cannot parse {val} to float")
    raise ValueError("Not a valid type for float conversion")

# --- Evaluators ---


def format_reward_func(completions):
    return [1.0 if extract_json(comp) is not None else -2.0 for comp in completions]


def schema_enforcement_reward_func(completions):
    rewards = []
    required_keys = {"title", "panel_count",
                     "panel_layout", "panels", "chart_type"}
    forbidden_keys = {"math", "rel", "stats", "trend"}

    for comp in completions:
        parsed = extract_json(comp)
        if not parsed:
            rewards.append(0.0)
            continue
        score = 0.5 if required_keys.issubset(parsed.keys()) else 0.0
        if any(f'"{fk}"' in str(comp).lower() for fk in forbidden_keys):
            score -= 1.0
        rewards.append(score)
    return rewards


def panel_architecture_reward_func(completions):
    rewards = []
    for comp in completions:
        parsed = extract_json(comp)
        if not parsed or "panels" not in parsed or "panel_count" not in parsed:
            rewards.append(0.0)
            continue
        score = 0.0
        try:
            actual_count = len(parsed["panels"])
            if actual_count == parsed["panel_count"]:
                score += 0.5
            if "panel_layout" in parsed and len(parsed["panel_layout"]) == 2:
                if (parsed["panel_layout"][0] * parsed["panel_layout"][1]) >= actual_count:
                    score += 0.5
        except (TypeError, ValueError):
            pass
        rewards.append(score)
    return rewards


def axis_mapping_reward_func(completions):
    rewards = []
    for comp in completions:
        parsed = extract_json(comp)
        if not parsed or "panels" not in parsed:
            rewards.append(0.0)
            continue
        score, valid_panels = 0.0, 0
        for panel in parsed["panels"]:
            if "axes" in panel and isinstance(panel["axes"], list):
                axes_names = [ax.get("name")
                              for ax in panel["axes"] if isinstance(ax, dict)]
                if "x_axis" in axes_names and "y_axis" in axes_names:
                    valid_panels += 1
        if len(parsed["panels"]) > 0:
            score += (valid_panels / len(parsed["panels"])) * 1.0
        rewards.append(score)
    return rewards

# NOTE: Add your full `evaluate_bar_series`, `evaluate_line_series`, etc. here if you want data scoring!


def dynamic_series_reward_func(completions, target_specs):
    # Dummy implementation for script brevity - replace with your full routing logic
    return [0.0 for _ in completions]

# ==========================================
# 3. DATA LOADING
# ==========================================


def load_eval_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    for item in data:
        raw_img_path = item["images"][0].replace("\\", "/")
        img_filename = os.path.basename(raw_img_path)
        local_img_path = os.path.join(LOCAL_IMAGE_DIR, img_filename)

        # Determine the target spec string
        target_spec_str = item["messages"][1]["content"] if len(
            item["messages"]) > 1 else ""

        processed_data.append({
            "id": str(item.get("id", "unknown")),
            "messages": item["messages"],
            "images": [local_img_path],
            "target_specs": target_spec_str
        })
    return Dataset.from_list(processed_data)

# ==========================================
# 4. MAIN INFERENCE LOOP
# ==========================================


def main():
    print("Loading dataset...")
    eval_dataset = load_eval_dataset(TEST_DATASET_PATH)
    print(f"Loaded {len(eval_dataset)} samples.")

    print("Loading model in 4-bit precision for RTX 3050 Mobile...")
    model, processor = FastVisionModel.from_pretrained(
        BASE_MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    print("Applying LoRA adapter...")
    model.load_adapter(LORA_ADAPTER_PATH)
    FastVisionModel.for_inference(model)
    processor.image_processor.max_pixels = 262144  # 512x512 limit for 4GB/6GB VRAM

    all_completions = []
    all_targets = []

    print("Running Inference...")
    for item in tqdm(eval_dataset):
        prompt = processor.apply_chat_template(
            [item["messages"][0]], tokenize=False, add_generation_prompt=True)

        try:
            pil_image = PIL.Image.open(item["images"][0]).convert("RGB")
        except Exception as e:
            print(f"Skipping image {item['images'][0]}: {e}")
            continue

        inputs = processor(
            text=[prompt],
            images=[pil_image],
            return_tensors="pt",
            padding=True
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1500,
                temperature=0.0
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        completion = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        all_completions.append(completion)
        all_targets.append(item["target_specs"])

    print("\n=== EVALUATION RESULTS ===")
    format_scores = format_reward_func(all_completions)
    print(
        f"Format Accuracy:         {sum(1 for s in format_scores if s > 0) / len(format_scores) * 100:.2f}%")

    schema_scores = schema_enforcement_reward_func(all_completions)
    print(
        f"Avg Schema Reward:       {sum(schema_scores)/len(schema_scores):.4f}")

    arch_scores = panel_architecture_reward_func(all_completions)
    print(f"Avg Architecture Reward: {sum(arch_scores)/len(arch_scores):.4f}")

    axis_scores = axis_mapping_reward_func(all_completions)
    print(f"Avg Axis Reward:         {sum(axis_scores)/len(axis_scores):.4f}")

    data_scores = dynamic_series_reward_func(all_completions, all_targets)
    print(
        f"Avg Series Data Reward:  {sum(data_scores)/max(1, len(data_scores)):.4f}")


if __name__ == "__main__":
    main()
