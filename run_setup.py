# -*- coding: utf-8 -*-
"""
Charles PA-DLLM Script

Migrated from Google Colab to local server environment.

# Basics

Accounts
"""

# Set HuggingFace cache directories to local project folder
import os
os.environ["HF_HOME"] = "./cache"
os.environ["HF_DATASETS_CACHE"] = "./cache/datasets"

# Google Drive mounting removed - running on local server

# log in to hf to use gated models
from huggingface_hub import login
import os

# Get token from environment variable (safer)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HUGGINGFACE_TOKEN:
    login(token=HUGGINGFACE_TOKEN)
else:
    print("Warning: HUGGINGFACE_TOKEN not set. Set it with: export HUGGINGFACE_TOKEN=your_token_here")
    print("Get your token from: https://huggingface.co/settings/tokens")

# log in to wandb
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY
    os.environ["WANDB_MODE"] = "online"
else:
    print("Warning: WANDB_API_KEY not set. Set it with: export WANDB_API_KEY=your_key_here")
# !wandb login --relogin

"""Packages"""

# Required packages (install if not already available):
# pip install transformers accelerate trl peft evaluate datasets

# Import
import pandas as pd
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoConfig,
    BitsAndBytesConfig
)
from peft import PeftModel

# Git repository setup (already done on server)

"""# Inference"""

# Google Drive refresh removed - running on local server

# Imports
import importlib

import inference
importlib.reload(inference)
import generate
importlib.reload(generate)

# Load inference functions
from inference import load_model, run_inference, run_greedy_inference, run_inference_batch, InferenceArguments, load_gsm8k, calculate_score
from generate import generate_vanilla

# Load gsm8k
n = 10 # number of questions to load
load_gsm8k(n)

# Load Model

from trl import ScriptArguments, ModelConfig

# Pass HF args
script_args = ScriptArguments(
    dataset_name=None,
    dataset_train_split=None,
    dataset_test_split=None,
)

model_args = ModelConfig(
    model_name_or_path="GSAI-ML/LLaDA-8B-Instruct",
    trust_remote_code=True,
    torch_dtype="auto",
)

# Pass custom inference args
inference_args = InferenceArguments(
    eval_json_path=None,
    adapter_path=None,
    output_csv_path=None,
    use_quantization=True,
    use_bitwise_lora=True,
)

# Load model
model, tokenizer, device = load_model(model_args)

# FASTEST: Save only model weights and essential config
torch.save(model.state_dict(), './cache/model_weights.pt')
tokenizer.save_pretrained('./cache/tokenizer/')

# Save only essential string values (safer)
config_data = {
    'model_name_or_path': model_args.model_name_or_path,
    'trust_remote_code': model_args.trust_remote_code,
    'torch_dtype': 'bfloat16',  # Use bfloat16 like original load_model
    'device': str(device)
}
torch.save(config_data, './cache/model_config.pt')

print("✅ Model saved (weights only) - fastest loading method!")