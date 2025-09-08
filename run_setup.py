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
# HUGGINGFACE_TOKEN = "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" # read
HUGGINGFACE_TOKEN = "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" # write
login(token=HUGGINGFACE_TOKEN)

# log in to wandb
import os
os.environ["WANDB_API_KEY"] = "9fc1866aae84e18298ea6bf1d417a104d21e6168"
os.environ["WANDB_MODE"] = "online"
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