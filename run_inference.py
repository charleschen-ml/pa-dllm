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
import time

# Imports
import inference
importlib = __import__('importlib')
importlib.reload(inference)

# Load inference functions
from inference import run_inference_batch, calculate_score, run_greedy_inference, run_inference, generate_one_sample
from inference import augment_one_sample, load_gsm8k, augment_multiple_samples
from inference import augment_one_sample_batch
from generate import generate_vanilla, generate_custom

# FASTEST: Load model weights and recreate architecture
print("Loading saved model (fast method)...")

# Set memory management environment variables
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

if __name__ == '__main__':
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
    # Instruction to prepend to each question
    ########################################################
    instruction = "Solve this problem and put your final answer in \\boxed{}:\n"
# instruction = """
# Solve this problem. Use this format:
# Reasoning: <reasoning here>
# Final Answer: \\boxed{<number>}
# """
# instruction = None

########################################################
# Create dataset of questions answered correctly
########################################################
# Load gsm8k
# df = load_gsm8k(10)

# Run batch inference
# df = run_inference_batch(
#     model=model,
#     tokenizer=tokenizer,
#     device=device,
#     model_args=model_args,
#     input_csv_path="./data/gsm8k.csv",
#     output_csv_path="./data/gsm8k_output.csv",
#     steps=128,
#     gen_length=128,
#     block_length=1,
#     instruction=instruction
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

# question = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?\n"
# correct_answer = 72

    question = df.iloc[0]['question'] # load the first question in df
    correct_answer = int(df.iloc[0]['answer_numerical'])  # extract the correct numerical answer

    if instruction is not None:
        question = instruction + question
    prompt = question

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
# print("🚀 Starting generate_one_sample...")
# start_time = time.time()
# manual_settings = {}
# training_sample = generate_one_sample(
#     model, tokenizer, device, prompt, model_args, 
#     gen_length=128, 
#     base_block_length=1, 
#     steps=128, 
#     curr_pos=0, 
#     correct_answer=correct_answer,
#     manual_settings=manual_settings,)
# print(f"training_sample=\n{training_sample}")
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"\n⏱️  TIMING REPORT:")
# print(f"  ⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")

########################################################
# Augment one sample (COMMENTED OUT - using parallel version below)
########################################################
# print("🚀 Starting augment_multiple_samples...")
# start_time = time.time()
# training_samples = augment_one_sample(
#     model=model,
#     tokenizer=tokenizer,
#     device=device,
#     prompt=prompt,
#     model_args=model_args,
#     gen_length=32,
#     base_block_length=1,
#     steps=32,
#     correct_answer=correct_answer,
#     break_after_answer_found=True  # Set to False to continue augmentation after answer found
# )
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"\n⏱️  TIMING REPORT:")
# print(f"  ⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")

########################################################
# Augment one sample (BATCHED)
########################################################
# print("🚀 Starting augment_multiple_samples...")
# start_time = time.time()
# training_samples_batch = augment_one_sample_batch(
#     model=model,
#     tokenizer=tokenizer,
#     device=device,
#     prompt=prompt,
#     model_args=model_args,
#     gen_length=32,
#     base_block_length=1,
#     steps=32,
#     correct_answer=correct_answer,
#     break_after_answer_found=True
# )
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"🚀 Batch augmentation produced {len(training_samples_batch)} samples")
# print(f"\n⏱️  TIMING REPORT:")
# print(f"  ⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")

########################################################
# Augment multiple samples
########################################################
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
    instruction=instruction
)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"\n⏱️  TIMING REPORT:")
print(f"  📊 Total samples generated: {len(all_training_samples)}")
print(f"  ⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
print(f"  ⚡ Time per sample: {elapsed_time/len(all_training_samples):.2f} seconds")

########################################################
# Augment multiple samples (PARALLEL - 2 GPUs)
########################################################
# print("🚀 Starting augment_multiple_samples_parallel with 2 GPUs...")
# start_time = time.time()

# from inference import augment_multiple_samples_parallel

# all_training_samples = augment_multiple_samples_parallel(
#     model_args=model_args,
#     csv_path="./data/gsm8k_correct.csv",
#     num_questions=2,  # Will split 1 questions per GPU
#     gen_length=32,
#     base_block_length=1,
#     steps=32,
#     break_after_answer_found=True,
#     output_json_path="./data/sft_training_samples_multi_greedy_parallel.json",
#     output_csv_path="./data/sft_training_samples_multi_greedy_parallel.csv",
#     instruction=instruction,
#     num_gpus=2  # Use 2 GPUs
# )

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"\n⏱️  TIMING REPORT:")
# print(f"  📊 Total samples generated: {len(all_training_samples)}")
# print(f"  ⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
# print(f"  ⚡ Time per sample: {elapsed_time/len(all_training_samples):.2f} seconds")
# print(f"  🚀 Used 2 GPUs in parallel!")
# print(f"  🎯 Processing rate: {len(all_training_samples)/elapsed_time:.1f} samples/second")