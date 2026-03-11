This is a fantastic milestone. Having that micro-batch pipeline working end-to-end guarantees you will have solid, concrete preliminary results to show off at your midterm presentation next month, regardless of how the full training run goes.

Now that the adapter is saved, you need to load it back up and test it on an unseen chart to verify it's actually extracting the `Extended ChartSpec` and not just hallucinating.

Here is the exact inference script and the game plan for your full dataset.

### Part 1: The Inference Script

To run inference, you have to load the base Qwen model exactly as you did during training (in 4-bit, skipping the vision tower), and then attach your saved LoRA adapter weights on top of it.

Create a new Jupyter notebook in your `/workspace/` and run this:

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# 1. Setup paths
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
adapter_path = "/workspace/qwen_lora_final"
test_image_path = "/workspace/balanced_images/UNSEEN_CHART_ID.png" # Pick one NOT in your 300 batch

# 2. Load Processor and Base Model
processor = AutoProcessor.from_pretrained(model_id)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["visual"] 
)

print("Loading base model...")
base_model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 3. Attach your trained LoRA adapter
print("Attaching LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval() # Crucial: set to evaluation mode

# 4. Prepare the test prompt
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": test_image_path},
            {"type": "text", "text": "Extract the Extended ChartSpec from this chart image. Return only the valid JSON."}
        ]
    }
]

# Apply chat template and process vision inputs
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info([messages])

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to("cuda")

# 5. Generate the JSON
print("Generating Extended ChartSpec...")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=1500, # Give it enough room for the full JSON
        temperature=0.1,     # Keep it low for structured JSON tasks
        do_sample=False      # Greedy decoding is best for exact formatting
    )

# 6. Clean up and print the output
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_json = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print("\n=== MODEL OUTPUT ===")
print(output_json)

```

### Part 2: Scaling to the Full 2,581 Dataset

When you transition from 300 samples to your sampled-down 2,581 dataset, the mechanics of the code are identical. However, to get the best performance over a longer run, you need to swap out a few hardcoded values in your `SFTConfig` for dynamic training hyperparameters.

Here are the key changes you need to make to your config block:

1. **Drop `max_steps` for `num_train_epochs`:** Instead of hardcoding 100 steps, tell the trainer to run through your entire dataset a specific number of times. For 2,581 samples, 2 to 3 epochs is the sweet spot.
2. **Add a Learning Rate Scheduler:** In a longer run, keeping the learning rate static at `2e-4` can cause the model to overshoot optimal weights near the end. You want a "cosine" scheduler that starts strong and gently tapers off.
3. **Add Warmup Steps:** Let the model ease into the new dataset for the first 5% of training before hitting the peak learning rate.

Update your `SFTConfig` for the full run to look like this:

```python
training_args = SFTConfig(
    output_dir="/workspace/qwen_lora_full_run",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    
    # --- NEW SCALING HYPERPARAMETERS ---
    num_train_epochs=2,              # Run through the 2581 samples twice
    lr_scheduler_type="cosine",      # Gently decay the learning rate
    warmup_ratio=0.05,               # Warmup for the first 5% of steps
    save_steps=50,                   # Save checkpoint every 50 steps (for Spot instance safety)
    # -----------------------------------
    
    save_total_limit=3,
    bf16=True,
    fp16=False,
    logging_steps=10,
    optim="paged_adamw_8bit",
    max_length=2048,
    remove_unused_columns=False,
    gradient_checkpointing=True
)

```

**RunPod Strategy for the Full Run:**
Because this will take significantly longer, this is exactly when you want to deploy a **Spot (Interruptible) Instance** to save your budget. An RTX 4090 or RTX A5000 is still perfect for this. Thanks to your `resume_from_checkpoint=True` setup, if the Spot instance drops, you just deploy a new one, hit run, and it picks up right where it left off.

Would you like to draft a quick validation script to automatically parse the model's output through Python's `json` library, just to mathematically verify that the full run is producing 100% syntactically valid JSONs?