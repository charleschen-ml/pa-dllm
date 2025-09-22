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
from inference import augment_one_sample, load_gsm8k
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
import pandas as pd
import json
import csv

# Load correct questions
correct_path = "./data/gsm8k_correct.csv"
df = pd.read_csv(correct_path)

# Truncate to desired number of questions
num_questions = 2  # Change this to any number you want
df = df.head(num_questions)
print(f"Processing {len(df)} questions (truncated from full dataset)")

# Parameters
instr = "Solve this problem and box your final answer:\n"
gen_length = 32
base_block_length = 1
steps = 32

# Collect all training samples from all questions
all_training_samples = []

print(f"Processing {len(df)} questions...")

for i in range(len(df)):
    question = df.iloc[i]['question']
    correct_answer = int(df.iloc[i]['answer_numerical'])  # Extract numerical answer
    prompt = instr + question
    
    print(f"\n{'='*60}")
    print(f"Processing question {i+1}/{len(df)}")
    print(f"Question: {question[:100]}...")
    print(f"Correct answer: {correct_answer}")
    print(f"{'='*60}")
    
    # Generate training samples for this question (but don't save files yet)
    question_samples = augment_one_sample(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        model_args=model_args,
        gen_length=gen_length,
        base_block_length=base_block_length,
        steps=steps,
        correct_answer=correct_answer,
        break_after_answer_found=True  # Add the new flag
    )
    
    # Add question metadata to each sample
    for sample in question_samples:
        sample['question_id'] = i
        sample['question'] = question
    
    all_training_samples.extend(question_samples)
    print(f"Generated {len(question_samples)} samples for question {i+1}")

# Save all samples to files
print(f"\n{'='*60}")
print(f"Saving {len(all_training_samples)} total training samples...")

# Save to JSON file
json_output_path = "./data/sft_training_samples_multi_greedy.json"
with open(json_output_path, "w") as f:
    json.dump(all_training_samples, f, indent=2)

# Save to CSV file for easier review (same format as augment_one_sample)
csv_output_path = "./data/sft_training_samples_multi_greedy.csv"
with open(csv_output_path, "w", newline='') as f:
    writer = csv.writer(f)
    
    # Write header (same as augment_one_sample)
    header = [
        'sample_id', 'confidence', 'entropy', 'position', 'token_id', 'token_text', 
        'position_relative', 'conf_0', 'entropy_0', 'top1_margin', 'mean_confidence', 'mean_entropy',
        'conf_std', 'entropy_std', 'conf_1', 'top4_conf_min', 'next4_conf_min',
        'top8_conf_min', 'next8_conf_min', 'block_size', 'answer_found',
        'full_confidence_list', 'full_entropy_list', 'full_token_ids', 'full_token_texts',
        'question_id', 'question'  # Add question metadata at the end
    ]
    writer.writerow(header)
    
    # Write data (same format as augment_one_sample)
    for sample_id, sample in enumerate(all_training_samples):
        block_size = sample['block_size']
        question_id = sample['question_id']
        question = sample['question']
        
        # Process each feature row (same logic as augment_one_sample)
        features = sample.get('features', [])
        if features:
            for feature in features:
                if len(feature) >= 13:  # Full feature set
                    confidence, entropy, position, token_id, token_text, position_relative = feature[:6]
                    (conf_0, entropy_0, top1_margin, mean_confidence, mean_entropy,
                     conf_std, entropy_std, conf_1, top4_conf_min, next4_conf_min,
                     top8_conf_min, next8_conf_min) = feature[6:18] if len(feature) >= 18 else feature[6:6+12]
                else:
                    # Handle incomplete features
                    confidence, entropy, position = feature[:3]
                    token_id, token_text, position_relative = feature[3:6] if len(feature) > 3 else [None, None, 0.0]
                    (conf_0, entropy_0, top1_margin, mean_confidence, mean_entropy,
                     conf_std, entropy_std, conf_1, top4_conf_min, next4_conf_min,
                     top8_conf_min, next8_conf_min) = [0.0] * 12
                
                writer.writerow([
                    sample_id, round(confidence, 4), round(entropy, 4), position, token_id, token_text,
                    round(position_relative, 4), round(conf_0, 4), round(entropy_0, 4), round(top1_margin, 4), 
                    round(mean_confidence, 4), round(mean_entropy, 4),
                    round(conf_std, 4), round(entropy_std, 4), round(conf_1, 4), 
                    round(top4_conf_min, 4), round(next4_conf_min, 4),
                    round(top8_conf_min, 4), round(next8_conf_min, 4), block_size, sample.get('answer_found', False),
                    sample.get('full_confidence_list', []),
                    sample.get('full_entropy_list', []),
                    sample.get('full_token_ids', []),
                    sample.get('full_token_texts', []),
                    question_id, question
                ])
        else:
            # No features available, write a default row
            writer.writerow([sample_id, 0.0, 0.0, 0, None, None, 0.0] + [0.0] * 12 + [1, False, [], [], [], [], question_id, question])

print(f"\n✅ Done. Saved {len(all_training_samples)} samples to:")
print(f"  📄 JSON: {json_output_path}")
print(f"  📊 CSV:  {csv_output_path}")
print(f"  📈 Samples per question: {len(all_training_samples) / len(df):.1f} avg")

