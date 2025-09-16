# -*- coding: utf-8 -*-
"""
Run Inference Script - Load saved model and run inference
"""

# Set HuggingFace cache directories to local project folder
import os
os.environ["HF_HOME"] = "./cache"
os.environ["HF_DATASETS_CACHE"] = "./cache/datasets"

# Import
import pandas as pd
import torch

# Imports
import inference
importlib = __import__('importlib')
importlib.reload(inference)

# Load inference functions
from inference import run_inference_batch, calculate_score, run_greedy_inference, run_inference, generate_one_sample
from inference import augment_one_sample
from generate import generate_vanilla, generate_custom

# FASTEST: Load model weights and recreate architecture
print("Loading saved model (fast method)...")

# Set memory management environment variables
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Load simple config (safer)
from trl import ModelConfig
config = torch.load('./cache/model_config.pt', weights_only=True)

# Recreate ModelConfig from saved values
model_args = ModelConfig(
    model_name_or_path=config['model_name_or_path'],
    trust_remote_code=config['trust_remote_code'],
    torch_dtype='bfloat16'  # Use bfloat16 like original
)
device_str = config['device']

# Recreate model architecture (fast) - use same method as original
from transformers import AutoModel, AutoTokenizer

# Clear any existing GPU memory first
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture on CPU first to avoid memory conflicts
model = AutoModel.from_pretrained(
    model_args.model_name_or_path,
    trust_remote_code=model_args.trust_remote_code,
    torch_dtype=torch.bfloat16,  # Use bfloat16 like original
    device_map=None,  # Load on CPU first
    low_cpu_mem_usage=True
)

# Load saved weights (fast) - keep on CPU
state_dict = torch.load('./cache/model_weights.pt', weights_only=True, map_location='cpu')

# Load state_dict while model is still on CPU
model.load_state_dict(state_dict)

# Now move to GPU if available (after weights are loaded)
model = model.to(device).eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('./cache/tokenizer/')

print(f"✅ Loaded model: {model_args.model_name_or_path} on {device}")

# Memory usage info
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(".1f")
    print(".1f")

# Run single inference
# run_inference(model, tokenizer, device, prompt, model_args, gen_length=16, base_block_length=2, steps=8)

# """Batch"""

# # Google Drive refresh removed - running on local server

# # Run batch inference
# df = run_inference_batch(
#     model=model,
#     tokenizer=tokenizer,
#     device=device,
#     model_args=model_args,
#     input_csv_path="./data/gsm8k.csv",
#     output_csv_path="./data/gsm8k_output.csv",
#     steps=32,
#     gen_length=32,
#     block_length=1
# )

# # Load df from csv
# df = pd.read_csv("./data/gsm8k_output.csv")

# # Google Drive refresh removed - running on local server

# # Calculate score
# correct_path = "./data/gsm8k_correct.csv"
# calculate_score(df, correct_path)

"""Greedy"""

# Load single prompt
instr = "Solve this problem and box your final answer:\n"
question = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?\n"
prompt = instr + question

########################################################
# Run single inference
########################################################
# run_inference(model, tokenizer, device, prompt, model_args, gen_length=32, base_block_length=1, steps=32)

########################################################
# Run greedy inference
########################################################
# run_greedy_inference(model, tokenizer, device, prompt, model_args, gen_length=16, base_block_length=1, steps=16)

########################################################
# Generate one sample
########################################################
# manual_settings = {
#     0:1,}
# training_sample = generate_one_sample(
#     model, tokenizer, device, prompt, model_args, 
#     gen_length=16, 
#     base_block_length=1, 
#     steps=16, 
#     curr_pos=1, 
#     manual_settings=manual_settings,)
# print(f"training_sample=\n{training_sample}")

########################################################
# Augment one sample
########################################################
# Params
gen_length = 16
base_block_length = 1
steps = 16
training_samples = augment_one_sample(
    model=model,
    tokenizer=tokenizer,
    device=device,
    prompt=prompt,
    model_args=model_args,
    gen_length=gen_length,
    base_block_length=base_block_length,
    steps=steps
)