# -*- coding: utf-8 -*-
"""
Run Inference Script - Load saved model and run inference
"""

# Set HuggingFace cache directories to local project cache
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
from inference import augment_one_sample, load_gsm8k, augment_multiple_samples
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

# Set device - use the available GPU (H100 is on cuda:0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model architecture first (empty model)
print("Loading model architecture...")
model = AutoModel.from_pretrained(
    model_args.model_name_or_path,
    trust_remote_code=model_args.trust_remote_code,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)
print("✅ Model architecture loaded")

# Load saved weights (much faster than downloading)
print("Loading saved model weights...")
state_dict = torch.load('./cache/model_weights.pt', weights_only=True, map_location='cpu')
model.load_state_dict(state_dict)

# Now move to GPU
model = model.to(device).eval()
print("✅ Model loaded from saved weights (fast)")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('./cache/tokenizer/')

print(f"✅ Loaded model: {model_args.model_name_or_path} on {device}")
print(f"📁 Cache location: ./cache/")

# Memory usage info
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")

########################################################
# Create dataset of questions answered correctly
########################################################
# # Load gsm8k
# df = load_gsm8k(10)

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
# # Calculate score
# correct_path = "./data/gsm8k_correct.csv"
# calculate_score(df, correct_path)

########################################################
# Load single prompt
########################################################
# df = pd.read_csv("./data/gsm8k_correct.csv")
# instr = "Solve this problem and box your final answer:\n"

# question = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?\n"
# correct_answer = 72

# # question = df.iloc[1]['question'] # load the first question in df
# # correct_answer = int(df.iloc[1]['answer_numerical'])  # extract the correct numerical answer

# prompt = instr + question

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
# manual_settings = {}
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
# gen_length = 16
# base_block_length = 1
# steps = 16
# training_samples = augment_one_sample(
#     model=model,
#     tokenizer=tokenizer,
#     device=device,
#     prompt=prompt,
#     model_args=model_args,
#     gen_length=gen_length,
#     base_block_length=base_block_length,
#     steps=steps,
#     correct_answer=correct_answer,
#     break_after_answer_found=True  # Set to False to continue augmentation after answer found
# )

########################################################
# Augment multiple samples
########################################################
import time
print("🚀 Starting augment_multiple_samples...")
start_time = time.time()

all_training_samples = augment_multiple_samples(
    model=model,
    tokenizer=tokenizer,
    device=device,
    model_args=model_args,
    csv_path="./data/gsm8k_correct.csv",
    num_questions=1,  # Change this to any number you want
    gen_length=32,
    base_block_length=1,
    steps=32,
    break_after_answer_found=True,
    output_json_path="./data/sft_training_samples_multi_greedy.json",
    output_csv_path="./data/sft_training_samples_multi_greedy.csv",
    verbose=False  # Set to True to see detailed progress logs
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n⏱️  TIMING REPORT:")
print(f"  📊 Total samples generated: {len(all_training_samples)}")
print(f"  ⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
print(f"  ⚡ Time per sample: {elapsed_time/len(all_training_samples):.2f} seconds")
print(f"  🎯 Processing rate: {len(all_training_samples)/elapsed_time:.1f} samples/second")

