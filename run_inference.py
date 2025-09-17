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

# Set device - use GPU 1 instead of 0 (GPU 0 is being used by another user)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

########################################################
# Create dataset of questions answered correctly
########################################################
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
df = pd.read_csv("./data/gsm8k_correct.csv")
instr = "Solve this problem and box your final answer:\n"

question = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?\n"
correct_answer = 72

# question = df.iloc[0]['question'] # load the first question in df
# correct_answer = int(df.iloc[0]['answer_numerical'])  # extract the correct numerical answer

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
    steps=steps,
    correct_answer=correct_answer
)

########################################################
# Augment multiple samples
########################################################
# import pandas as pd
# import json
# import csv

# # Load correct questions
# correct_path = "./data/gsm8k_correct.csv"
# df = pd.read_csv(correct_path)

# # Truncate to desired number of questions
# num_questions = 2  # Change this to any number you want
# df = df.head(num_questions)
# print(f"Processing {len(df)} questions (truncated from full dataset)")

# # Parameters
# instr = "Solve this problem and box your final answer:\n"
# gen_length = 16
# base_block_length = 1
# steps = 16

# # Collect all training samples from all questions
# all_training_samples = []

# print(f"Processing {len(df)} questions...")

# for i in range(len(df)):
#     question = df.iloc[i]['question']
#     correct_answer = int(df.iloc[i]['answer_numerical'])  # Extract numerical answer
#     prompt = instr + question
    
#     print(f"\n{'='*60}")
#     print(f"Processing question {i+1}/{len(df)}")
#     print(f"Question: {question[:100]}...")
#     print(f"Correct answer: {correct_answer}")
#     print(f"{'='*60}")
    
#     # Generate training samples for this question (but don't save files yet)
#     question_samples = augment_one_sample(
#         model=model,
#         tokenizer=tokenizer,
#         device=device,
#         prompt=prompt,
#         model_args=model_args,
#         gen_length=gen_length,
#         base_block_length=base_block_length,
#         steps=steps,
#         save_files=False,  # Don't save files for each question
#         correct_answer=correct_answer  # Pass the correct answer
#     )
    
#     # Add question metadata to each sample
#     for sample in question_samples:
#         sample['question_id'] = i
#         sample['question'] = question
    
#     all_training_samples.extend(question_samples)
#     print(f"Generated {len(question_samples)} samples for question {i+1}")

# # Save all samples to files
# print(f"\n{'='*60}")
# print(f"Saving {len(all_training_samples)} total training samples...")

# # Save to JSON file
# json_output_path = "./data/sft_training_samples_multi_greedy.json"
# with open(json_output_path, "w") as f:
#     json.dump(all_training_samples, f, indent=2)

# # Save to CSV file for easier review
# csv_output_path = "./data/sft_training_samples_multi_greedy.csv"
# with open(csv_output_path, "w", newline='') as f:
#     writer = csv.writer(f)
    
#     # Write header
#     writer.writerow(['sample_id', 'question_id', 'confidence', 'entropy', 'position', 'block_size'])
    
#     # Write data
#     for sample_id, sample in enumerate(all_training_samples):
#         block_size = sample['block_size']
#         question_id = sample['question_id']
#         for feature in sample['features']:
#             confidence, entropy, position = feature
#             writer.writerow([sample_id, question_id, confidence, entropy, position, block_size])

# print(f"\n✅ Done. Saved {len(all_training_samples)} samples to:")
# print(f"  📄 JSON: {json_output_path}")
# print(f"  📊 CSV:  {csv_output_path}")
# print(f"  📈 Samples per question: {len(all_training_samples) / len(df):.1f} avg")