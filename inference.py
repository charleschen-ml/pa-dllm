# Inference

import shutil
import evaluate
import json
import csv
import argparse
import re
import math
from math import ceil
from tqdm import tqdm
import torch
from peft import get_peft_model, PeftModel
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import (
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
# Ensure inference quantization config matches that of QAT
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # To fix torch deterministic error
torch.use_deterministic_algorithms(True)

from generate import generate_vanilla, generate_custom, extract_boxed # generate.py from llada github
from transformers import AutoTokenizer, AutoModel

# Custom arguments for inference-specific parameters
class InferenceArguments:
    def __init__(self, 
                eval_json_path="./data/eval_set.json",
                adapter_path="./data/gpt2-qat",
                output_csv_path="./data/inference_output.csv",
                bitwise_lora_adapter_path="./data/full_qat_model.pt",
                 use_quantization=True,
                 use_bitwise_lora=True,
                 bit_choices="32",
                 max_inf_size=100,
                 quant_layers="6,11",
                 inf_bit_config={},
                 default_bit=32):
        self.eval_json_path = eval_json_path
        self.adapter_path = adapter_path
        self.output_csv_path = output_csv_path
        self.bitwise_lora_adapter_path = bitwise_lora_adapter_path
        self.use_quantization = use_quantization
        self.use_bitwise_lora = use_bitwise_lora
        self.bit_choices = bit_choices
        self.max_inf_size = max_inf_size
        self.quant_layers = quant_layers
        self.inf_bit_config = inf_bit_config
        self.default_bit = default_bit

def load_gsm8k(n=100):
    from datasets import load_dataset
    import pandas as pd
    import re

    # Load GSM8K dataset (train split)
    ds = load_dataset("openai/gsm8k", "main", split="train")

    # Select the first n examples
    ds_small = ds.select(range(min(n, len(ds))))

    # Convert to pandas DataFrame
    df = pd.DataFrame(ds_small)

    # Extract the numerical answer from the last line starting with '####'
    def extract_numerical(answer):
        match = re.search(r"####\s*([\d.,]+)", answer)
        if match:
            num_str = match.group(1).replace(",", "")
            try:
                # If it's a whole number, return as int; otherwise float
                return int(num_str) if '.' not in num_str else float(num_str)
            except ValueError:
                return None
        return None

    df["answer_numerical"] = df["answer"].apply(extract_numerical)

    # Save to CSV
    output_path = "./data/gsm8k.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} examples to {output_path}")
    return df

def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("inference", help="Run the inference script", dataclass_types=dataclass_types)
    else:
        parser = HfArgumentParser(dataclass_types)
    
    # Add inference-specific arguments
    parser.add_argument("--eval_json_path", type=str, 
                       default="./data/eval_set.json",
                       help="Path to evaluation JSON file")
    parser.add_argument("--adapter_path", type=str,
                       default="./data/gpt2-qat",
                       help="Path to adapter directory")
    parser.add_argument("--output_csv_path", type=str,
                       default="./data/inference_output.csv",
                       help="Path to save inference output CSV")
    parser.add_argument("--bitwise_lora_adapter_path", type=str,
                       default="./data/full_qat_model.pt",
                       help="Path to bitwise LoRA adapter file")
    parser.add_argument("--use_quantization", action="store_true", default=True,
                       help="Whether to apply quantization")
    parser.add_argument("--no_quantization", dest="use_quantization", action="store_false",
                       help="Disable quantization")
    parser.add_argument("--use_bitwise_lora", action="store_true", default=True,
                       help="Whether to use bitwise LoRA adapters")
    parser.add_argument("--no_bitwise_lora", dest="use_bitwise_lora", action="store_false",
                       help="Disable bitwise LoRA adapters")
    parser.add_argument("--bit_choices", type=str, default="32",
                       help="Comma-separated list of bit choices for LoRA")
    parser.add_argument("--max_inf_size", type=int, default=100,
                       help="Maximum number of examples to infer")
    parser.add_argument("--quant_layers", type=str, default="6,11",
                       help="Comma-separated list of h.* layers to quantize")
    parser.add_argument("--inf_bit_config", type=str, default=None,
                       help="JSON string for inference bit configuration (e.g., '{\"transformer.h.0\": 8, \"transformer.h.1\": 4}'). Default: 32 bits for all layers")
    parser.add_argument("--default_bit", type=int, default=32,
                       help="Default bit for all layers")
    return parser

def calculate_block_sizes(gen_length, base_block_length, sweep_position=None, sweep_value=None, manual_settings=None):
    '''
    Calculate block sizes for sweeping experiments.
    
    Args:
        gen_length: Total generation length
        base_block_length: Base block length (e.g., 2)
        sweep_position: Which block to sweep (0-indexed) - for backward compatibility
        sweep_value: Value to set at sweep position - for backward compatibility
        manual_settings: Dict of {position: value} for manual settings (e.g., {0: 4, 1: 6})
    
    Returns:
        List of block sizes that sum to gen_length
        
    Example:
        calculate_block_sizes(32, 2, 0, 8) -> [8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        calculate_block_sizes(32, 2, manual_settings={0: 4, 1: 6}) -> [4, 6, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    '''
    # Calculate how many base blocks we need
    num_base_blocks = gen_length // base_block_length
    
    # Create list of base block sizes
    block_sizes = [base_block_length] * num_base_blocks
    
    # Handle manual settings
    if manual_settings is not None:
        # Convert old sweep parameters to manual_settings for backward compatibility
        if sweep_position is not None and sweep_value is not None:
            manual_settings = {sweep_position: sweep_value}
        
        # Validate all positions
        for pos in manual_settings.keys():
            if pos >= len(block_sizes):
                raise ValueError(f"Position {pos} must be less than number of blocks ({len(block_sizes)})")
        
        # Set all manual positions FIRST (skip None values)
        for pos, value in manual_settings.items():
            if value is not None:
                block_sizes[pos] = value
        
        # Calculate total adjustment needed (skip None values)
        total_adjustment = sum(value - base_block_length for value in manual_settings.values() if value is not None)
        
        # Check if adjustment is possible
        if total_adjustment > 0:
            # Need to reduce other blocks to accommodate larger manual blocks
            remaining_blocks = len(block_sizes) - len([v for v in manual_settings.values() if v is not None])
            max_reducible = remaining_blocks * base_block_length
            if total_adjustment > max_reducible:
                return None  # Signal that this configuration is not possible
            
            # Reduce blocks from the end: last block goes to 0 first, then second-to-last to 1, etc.
            blocks_to_reduce = total_adjustment
            j = len(block_sizes) - 1  # Start from last block
            while blocks_to_reduce > 0 and j >= 0:
                if j not in manual_settings or manual_settings[j] is None:
                    # Reduce this block completely before moving to the next
                    reduction = min(blocks_to_reduce, block_sizes[j])
                    block_sizes[j] -= reduction
                    blocks_to_reduce -= reduction
                j -= 1
            
            if blocks_to_reduce > 0:
                raise ValueError(f"Cannot accommodate manual settings with given parameters")
        
        elif total_adjustment < 0:
            # Need to add to other blocks when manual blocks are smaller
            blocks_to_add = -total_adjustment
            j = len(block_sizes) - 1  # Start from last block
            while blocks_to_add > 0 and j >= 0:
                if j not in manual_settings or manual_settings[j] is None:
                    # Add to this block
                    addition = min(blocks_to_add, gen_length - sum(block_sizes))
                    block_sizes[j] += addition
                    blocks_to_add -= addition
                j -= 1
    
    # Handle backward compatibility for old sweep parameters
    elif sweep_position is not None and sweep_value is not None:
        # Validate sweep position
        if sweep_position >= len(block_sizes):
            raise ValueError(f"sweep_position ({sweep_position}) must be less than number of blocks ({len(block_sizes)})")
        
        # Calculate the adjustment needed
        adjustment = sweep_value - base_block_length
        
        # Check if adjustment is possible
        if adjustment > 0:
            # Need to reduce other blocks to accommodate larger sweep block
            remaining_blocks = len(block_sizes) - 1
            if adjustment > remaining_blocks * base_block_length:
                raise ValueError(f"sweep_value ({sweep_value}) too large for gen_length ({gen_length})")
            
            # Reduce blocks from the end: last block goes to 0 first, then second-to-last to 1, etc.
            blocks_to_reduce = adjustment
            j = len(block_sizes) - 1  # Start from last block
            while blocks_to_reduce > 0 and j >= 0:
                if j != sweep_position:
                    # Reduce this block completely before moving to the next
                    reduction = min(blocks_to_reduce, block_sizes[j])
                    block_sizes[j] -= reduction
                    blocks_to_reduce -= reduction
                j -= 1
            
            if blocks_to_reduce > 0:
                raise ValueError(f"Cannot accommodate sweep_value ({sweep_value}) with given parameters")
        
        # Set the sweep position
        block_sizes[sweep_position] = sweep_value
    
    # Validate total
    total = sum(block_sizes)
    if total != gen_length:
        raise ValueError(f"Calculated block sizes sum to {total}, expected {gen_length}")
    
    # Keep zero elements to maintain consistent number of blocks
    # block_sizes = [size for size in block_sizes if size > 0]  # Commented out
    
    return block_sizes

# Load model
def load_model(model_args):
    """Load tokenizer + model once and return (model, tokenizer, device).
    Works for normal and k-bit quantized loads depending on model_args.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct', 
        trust_remote_code=True
    )

    print(f"✅ Loaded model: {model_args.model_name_or_path} on {device}")
    return model, tokenizer, device

# Run single inference
def run_inference(model, tokenizer, device, prompt, model_args, max_new_tokens=32, do_sample=False, gen_length=32, base_block_length=2, steps=16):
    """Run a single prompt without reloading the model.
    Pass model_args.use_cache=False for LLaDA/MDM-style models.
    """

    print(f"raw prompt=\n{prompt}")

    # Add special tokens for the Instruct model (not required for base model)
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    # prompt = prompt + "Lily can run \\boxed" # manually append golden tokens as an experiment 

    input_ids = tokenizer(prompt)['input_ids']

    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # debug
    inputs_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"templated prompt=\n{inputs_decoded}")

    # original llada generate()
    out = generate_vanilla(
        model, 
        tokenizer, # charles added
        input_ids, 
        steps=steps, 
        gen_length=gen_length, 
        block_length=base_block_length, 
        temperature=0., 
        cfg_scale=0., 
        remasking='low_confidence'
    )

    # debug: print raw decoded output
    print(f"raw output=\n{out[0]}")
    raw_decoded_output = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    print(f"raw decoded output=\n{raw_decoded_output}")

    # debug: print token_id: token mapping for the full output
    print(f"\nFull token breakdown:")
    for i, token_id in enumerate(out[0]):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        print(f"{token_id.item()}: '{token_text}'")

    out_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    print(f"out_text=\n{out_text}")
    # print(f"Answer correct? {extract_boxed(out_text) == 72}")

    # debug
    # outputs_ids = outputs[0]  # tensor of shape [1, seq_len]
    # print(f"outputs_ids=\n{outputs_ids}")
    # outputs_decoded = tokenizer.decode(outputs_ids, skip_special_tokens=False)
    # print(f"decoded output=\n{outputs_decoded}")

    return out # so we can go token by token

def run_inference_batch(model, tokenizer, device, model_args, input_csv_path, output_csv_path,
                        steps=16, gen_length=32, block_length=2):
    import pandas as pd
    import re

    def extract_numerical(text):
        """Extract the final boxed numerical answer (e.g., from \boxed{72}) or trailing number."""
        # Try to extract from \boxed{...}
        boxed_match = re.search(r"\\boxed{([\d.,]+)}", text)
        if boxed_match:
            num_str = boxed_match.group(1).replace(",", "")
        else:
            # Try to extract last number in string
            num_match = re.search(r"(\d+(?:\.\d+)?)\s*$", text.strip())
            num_str = num_match.group(1) if num_match else None

        if num_str is None:
            return None
        try:
            return int(num_str) if '.' not in num_str else float(num_str)
        except ValueError:
            return None

    print(f"📥 Loading: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    print(f"🔢 Found {len(df)} rows")

    completions = []
    completion_numericals = []

    for i, row in enumerate(df.itertuples(), start=1):
        question = getattr(row, "question")

        # Apply chat template
        instr = "Solve this problem and box your final answer:\n"
        messages = [{"role": "user", "content": instr + question}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # print decoded input
        print(f"decoded input=\n{tokenizer.decode(input_ids[0], skip_special_tokens=True)}")

        # Run generation
        out = generate_vanilla(
            model=model,
            tokenizer=tokenizer,
            prompt=input_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=0.0,
            cfg_scale=0.0,
            remasking="low_confidence"
        )

        # Decode the output
        output_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f"decoded output=\n{output_text}")
        numeric_answer = extract_numerical(output_text)

        completions.append(output_text)
        completion_numericals.append(numeric_answer)

        print(f"[{i}] ✅ Q: {question.strip()[:50]}... → A: {output_text.strip()[:50]}... → #{numeric_answer}")

    # Save results
    df["completion"] = completions
    df["completion_numerical"] = completion_numericals
    df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Saved output to: {output_csv_path}")

    return df

def calculate_score(df, correct_csv_path="correct_questions.csv"):
    """Calculate and print the percentage of correct answers, and save correct questions to a CSV."""
    correct_count = sum(df['answer_numerical'] == df['completion_numerical'])
    total_count = len(df)
    score_percentage = (correct_count / total_count) * 100
    print(f"Final Score: {correct_count}/{total_count} ({score_percentage:.2f}%) correct")

    # Filter correct questions
    correct_df = df[df['answer_numerical'] == df['completion_numerical']]
    correct_df.to_csv(correct_csv_path, index=False)
    print(f"Correct questions saved to {correct_csv_path}")

# Run Greedy inference
def run_greedy_inference(model, tokenizer, device, prompt, model_args, max_new_tokens=32, do_sample=False, gen_length=32, base_block_length=2, steps=16):
    """Run a single prompt without reloading the model.
    Pass model_args.use_cache=False for LLaDA/MDM-style models.
    Greedy search: optimize each position one by one
    """

    # Add special tokens for the Instruct model (not required for base model)
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']

    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # debug
    inputs_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"decoded inputs=\n{inputs_decoded}")

    # custom generate with block size as list
    first_correct_steps = []
    optimal_block_sizes = None
    optimal_output_text = None
    optimal_block_confidences = None
    min_step = float('inf')
    
    # Initialize manual_settings
    manual_settings = {}
    
    # Greedy search: optimize each position one by one
    num_blocks = gen_length // base_block_length
    for position in range(num_blocks):  # Search all positions
        print(f"\n=== Optimizing position {position} ===")
        best_value = None
        best_step = float('inf')
        
        for sweep_value in range(1, gen_length + 1):  # Try values 1 to gen_length
            current_settings = manual_settings.copy()
            current_settings[position] = sweep_value
            
            block_sizes = calculate_block_sizes(
                gen_length=gen_length, 
                base_block_length=base_block_length, 
                manual_settings=current_settings,
            )
            
            if block_sizes is None:
                continue
                
            print(f"Testing position {position} = {sweep_value}")
            
            out, first_correct_step, block_confidences, initial_entropy, initial_confidence, _, _ = generate_custom(
                model, 
                tokenizer, # charles added
                input_ids,
                steps=steps,
                gen_length=gen_length,
                block_sizes=block_sizes,
                temperature=0.,
                cfg_scale=0.,
                remasking='low_confidence'
            )
            first_correct_steps.append(first_correct_step)

            # Early exit: if no correct answer was ever found (inf), keep last good size and skip remaining sweeps
            if first_correct_step == float('inf'):
                if best_value is not None:
                    print(f"Found inf at position {position}, sweep_value {sweep_value}. Keeping last good size {best_value} and skipping remaining sweeps for this position.")
                    # best_step/best_value/optimal_* were already set when the good size was found
                else:
                    # No prior good result; record current outputs and exit
                    try:
                        optimal_output_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                    except Exception:
                        optimal_output_text = None
                    optimal_block_confidences = block_confidences.copy() if isinstance(block_confidences, dict) else block_confidences
                    optimal_block_sizes = block_sizes.copy()
                    best_step = first_correct_step
                    best_value = sweep_value
                    print(f"Found inf at position {position}, no prior good size. Skipping remaining sweeps for this position.")
                break

            # Track best value for current position
            # Prefer smaller first_correct_step; on ties, prefer larger sweep_value
            if (first_correct_step < best_step) or (
                first_correct_step == best_step and (best_value is None or sweep_value > best_value)
            ):
                best_step = first_correct_step
                best_value = sweep_value
                optimal_block_sizes = block_sizes.copy()
                # Store the output text and confidences for the best configuration
                optimal_output_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                optimal_block_confidences = block_confidences.copy()
        
        # Update manual_settings with the best value found for this position
        if best_value is not None:
            manual_settings[position] = best_value
            print(f"Best value for position {position}: {best_value} (step: {best_step})")
        else:
            print(f"No valid configuration found for position {position}")
        
        # Update global minimum
        if best_step < min_step:
            min_step = best_step
    
    # Print the list of first_correct_steps and find the minimum
    print(f"\nFirst correct steps: {first_correct_steps}")
    print(f"Minimum first correct step: {min_step}")
    print(f"Optimal block sizes: {optimal_block_sizes}")
    print(f"\nOptimal output text:")
    print(optimal_output_text)
    
    # Show detailed confidence tracking for the optimal configuration
    if optimal_block_sizes is not None:
        print(f"\n{'='*60}")
        print("DETAILED CONFIDENCE TRACKING")
        print(f"{'='*60}")

        # Generate the optimal configuration one more time to get the block breakdown
        optimal_settings = {}
        for i, size in enumerate(optimal_block_sizes):
            if size > 0:
                optimal_settings[i] = size

        final_block_sizes = calculate_block_sizes(
            gen_length=gen_length,
            base_block_length=base_block_length,
            manual_settings=optimal_settings,
        )

        if final_block_sizes is not None:
            # First, run generate_custom to get the final result and confidences
            out, _, final_confidences, _, _, _, _ = generate_custom(
                model,
                tokenizer,
                input_ids,
                steps=steps,
                gen_length=gen_length,
                block_sizes=final_block_sizes,
                temperature=0.,
                cfg_scale=0.,
                remasking='low_confidence'
            )


    # out_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    # print("\n" + out_text)
    # print(f"Answer correct? {extract_boxed(out_text) == 72}")

    # debug
    # outputs_ids = outputs[0]  # tensor of shape [1, seq_len]
    # print(f"outputs_ids=\n{outputs_ids}")
    # outputs_decoded = tokenizer.decode(outputs_ids, skip_special_tokens=False)
    # print(f"decoded output=\n{outputs_decoded}")

    return

# Generate one sample for SFT dataset
def generate_one_sample(model, tokenizer, device, prompt, model_args, max_new_tokens=32, do_sample=False, gen_length=32, base_block_length=2, steps=16, curr_pos=0, manual_settings=None, correct_answer=None):
    """Run a single prompt without reloading the model.
    Pass model_args.use_cache=False for LLaDA/MDM-style models.
    
    Args:
        manual_settings: Dict of {block_position: block_size_value} for manual block size settings.
                        If None, starts with empty dict (all blocks use base_block_length).
        correct_answer: Expected correct answer for checking correctness (optional).
    """

    # Add special tokens for the Instruct model (not required for base model)
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']

    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # debug
    inputs_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"decoded inputs=\n{inputs_decoded}")

    # custom generate with block size as list
    first_correct_steps = []
    optimal_block_sizes = None
    optimal_output_text = None
    optimal_block_confidences = None
    min_step = float('inf')
    
    # Initialize manual_settings
    if manual_settings is None:
        manual_settings = {}  # {block_position: block_size_value}
    
    # We'll use the confidence values from generate_custom instead of reimplementing
    print(f"\n=== Using confidence from generate_custom for curr_pos={curr_pos} ===")
    autoregressive_confidence = []
    autoregressive_entropy = []

    # Single token greedy search: optimize only the specified position
    num_blocks = gen_length // base_block_length
    position = curr_pos # optimize the position specified by curr_pos

    print(f"\n=== Optimizing position {position} (single token search) ===")
    best_value = None
    best_step = float('inf')
    optimal_out = None  # Store optimal token IDs
    stored_ar_tokens = None  # Store AR context tokens from first call
    stored_additional_features = None  # Store additional confidence metrics from first call
    
    for sweep_value in range(1, gen_length + 1):  # Try values 1 to gen_length
        current_settings = manual_settings.copy()
        current_settings[position] = sweep_value
        
        block_sizes = calculate_block_sizes(
            gen_length=gen_length, 
            base_block_length=base_block_length, 
            manual_settings=current_settings,
        )
        
        if block_sizes is None:
            continue
            
        print(f"Testing position {position} = {sweep_value}")
        
        out, first_correct_step, block_confidences, initial_entropy, initial_confidence, ar_context_tokens, additional_features = generate_custom(
            model, 
            tokenizer, # charles added
            input_ids,
            steps=steps,
            gen_length=gen_length,
            block_sizes=block_sizes,
            temperature=0.,
            cfg_scale=0.,
            remasking='low_confidence',
            curr_pos=curr_pos,  # Pass curr_pos to capture confidence/entropy at the right block
            correct_answer=correct_answer
        )
        first_correct_steps.append(first_correct_step)

        # Store AR context tokens, additional features, and confidence from the first call
        if stored_ar_tokens is None and ar_context_tokens is not None:
            stored_ar_tokens = ar_context_tokens.clone()
            stored_additional_features = additional_features.copy() if additional_features else {}
            
            # Extract the forward-looking confidence from initial_confidence
            if initial_confidence and len(initial_confidence) > curr_pos:
                autoregressive_confidence = initial_confidence[curr_pos:]
                autoregressive_entropy = initial_entropy[curr_pos:] if initial_entropy and len(initial_entropy) > curr_pos else []
                print(f"📝 Extracted {len(autoregressive_confidence)} forward-looking confidence values from curr_pos={curr_pos}")
                print(f"    Confidence values: {autoregressive_confidence[:5]}...")  # Show first 5
            else:
                autoregressive_confidence = []
                autoregressive_entropy = []
                print(f"⚠️ No confidence values available from generate_custom")
            
            print(f"📝 Stored AR context tokens, additional features, and confidence for analysis")

        # Early exit: if no correct answer was ever found (inf), keep last good size and skip remaining sweeps
        if first_correct_step == float('inf'):
            if best_value is not None:
                print(f"Found inf at position {position}, sweep_value {sweep_value}. Keeping last good size {best_value} and stopping search.")
                # best_step/best_value/optimal_* were already set when the good size was found
            else:
                # No prior good result; record current outputs and exit
                try:
                    optimal_output_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                except Exception:
                    optimal_output_text = None
                optimal_block_confidences = block_confidences.copy() if isinstance(block_confidences, dict) else block_confidences
                optimal_block_sizes = block_sizes.copy()
                best_step = first_correct_step
                best_value = sweep_value
                print(f"Found inf at position {position}, no prior good size. Stopping search.")
            break

        # Track best value for current position
        # Prefer smaller first_correct_step; on ties, prefer larger sweep_value
        if (first_correct_step < best_step) or (
            first_correct_step == best_step and (best_value is None or sweep_value > best_value)
        ):
            best_step = first_correct_step
            best_value = sweep_value
            optimal_block_sizes = block_sizes.copy()
            # Store the output text, token IDs, and confidences for the best configuration
            optimal_output_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
            optimal_out = out.clone()  # Store the token IDs
            optimal_block_confidences = block_confidences.copy()
    
    # Update manual_settings with the best value found for this position
    if best_value is not None:
        manual_settings[position] = best_value
        print(f"Best value for position {position}: {best_value} (step: {best_step})")
    else:
        print(f"No valid configuration found for position {position}")
    
    # Set global minimum to the best step found
    min_step = best_step
    
    # Print the list of first_correct_steps and find the minimum
    print(f"\nFirst correct steps: {first_correct_steps}")
    print(f"Minimum first correct step: {min_step}")
    print(f"Optimal block sizes: {optimal_block_sizes}")
    print(f"\nOptimal output text:")
    print(optimal_output_text)
    
    # Show detailed confidence tracking for the optimal configuration
    if optimal_block_sizes is not None:
        print(f"\n{'='*60}")
        print("DETAILED CONFIDENCE TRACKING")
        print(f"{'='*60}")

        # Generate the optimal configuration one more time to get the block breakdown
        optimal_settings = {}
        for i, size in enumerate(optimal_block_sizes):
            if size > 0:
                optimal_settings[i] = size

        final_block_sizes = calculate_block_sizes(
            gen_length=gen_length,
            base_block_length=base_block_length,
            manual_settings=optimal_settings,
        )

        if final_block_sizes is not None:
            # First, run generate_custom to get the final result and confidences
            out, _, final_confidences, _, _, _, _ = generate_custom(
                model,
                tokenizer,
                input_ids,
                steps=steps,
                gen_length=gen_length,
                block_sizes=final_block_sizes,
                temperature=0.,
                cfg_scale=0.,
                remasking='low_confidence'
            )

    # store the input-ouput pair in this format (e.g. curr_pos=4):
    # {
    #     "features": [
    #         [confidence, entropy, position, token_id, token_text, conf_0, entropy_0, top1_margin, 
    #          mean_confidence, mean_entropy, conf_std, entropy_std, conf_1, top4_conf_min, 
    #          next4_conf_min, top8_conf_min, next8_conf_min]  # ONE feature row (17 features total)
    #     ],
    #     "block_size": 2  # supervised label
    # }
    # 
    # NOTE: Features are captured AFTER AR context is built (blocks 0 to curr_pos-1)
    # - Basic: confidence, entropy, position, token_id, token_text
    # - Immediate: conf_0 (next token confidence), entropy_0, top1_margin
    # - Global: mean_confidence, mean_entropy, conf_std, entropy_std
    # - Specific: conf_1 (second token confidence)
    # - Comparative: top4/8_conf_min (best tokens), next4/8_conf_min (sequential tokens)
    # This allows analysis of which tokens are suitable for parallel vs sequential decoding.
    
    # Extract features for the current position curr_pos (only one feature row per curr_pos)
    features = []
    if len(autoregressive_confidence) > 0 and len(autoregressive_entropy) > 0:
        # The position we're analyzing is curr_pos (the current token to be decoded)
        position = curr_pos
        
        # In autoregressive_confidence, index 0 corresponds to position curr_pos
        confidence_idx = 0  # Index in autoregressive confidence list for curr_pos
        
        if confidence_idx < len(autoregressive_confidence) and confidence_idx < len(autoregressive_entropy):
            confidence = autoregressive_confidence[confidence_idx]
            entropy = autoregressive_entropy[confidence_idx]
            
            # Get token and token_id from AR context + top-1 predictions
            token_id = None
            token_text = None
            if stored_ar_tokens is not None:
                # Extract token_id from the AR context + top-1 predictions
                gen_start_idx = input_ids.shape[1]  # Start of generation
                token_pos_in_gen = curr_pos  # Position within generation
                if gen_start_idx + token_pos_in_gen < stored_ar_tokens.shape[1]:
                    token_id = stored_ar_tokens[0, gen_start_idx + token_pos_in_gen].item()
                    token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            
            # Get additional features for this position (calculated from curr_pos perspective)
            additional_feats = stored_additional_features.get(curr_pos, {}) if stored_additional_features else {}
            
            # Calculate position_relative (position / gen_length) with 4 decimal places
            position_relative = round(position / gen_length, 4)
            
            # Create comprehensive feature vector
            feature_row = [
                confidence, entropy, position, token_id, token_text,
                position_relative,
                additional_feats.get('conf_0', 0.0),
                additional_feats.get('entropy_0', 0.0), 
                additional_feats.get('top1_margin', 0.0),
                additional_feats.get('mean_confidence', 0.0),
                additional_feats.get('mean_entropy', 0.0),
                additional_feats.get('conf_std', 0.0),
                additional_feats.get('entropy_std', 0.0),
                additional_feats.get('conf_1', 0.0),
                additional_feats.get('top4_conf_min', 0.0),
                additional_feats.get('next4_conf_min', 0.0),
                additional_feats.get('top8_conf_min', 0.0),
                additional_feats.get('next8_conf_min', 0.0),
            ]
            
            features.append(feature_row)
        else:
            # Calculate position_relative for default row too
            position_relative = round(position / gen_length, 4)
            # Default feature row with zeros for missing data
            default_row = [0.0, 0.0, position, None, None, position_relative] + [0.0] * 12  # 12 additional features
            features.append(default_row)
    
    # Create training sample with full lists preserved
    full_token_ids = []
    full_token_texts = []
    
    if stored_ar_tokens is not None:
        gen_start_idx = input_ids.shape[1]
        gen_end_idx = gen_start_idx + gen_length
        for pos in range(gen_length):
            if gen_start_idx + pos < stored_ar_tokens.shape[1]:
                token_id = stored_ar_tokens[0, gen_start_idx + pos].item()
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                full_token_ids.append(token_id)
                full_token_texts.append(token_text)
            else:
                full_token_ids.append(None)
                full_token_texts.append(None)
    
    # Use the autoregressive confidence/entropy lists we calculated earlier
    training_sample = {
        "features": features,
        "block_size": best_value if best_value is not None else 1,
        "full_confidence_list": autoregressive_confidence,  # Forward-looking confidence from curr_pos
        "full_entropy_list": autoregressive_entropy,        # Forward-looking entropy from curr_pos
        "full_token_ids": full_token_ids,
        "full_token_texts": full_token_texts
    }
    
    print(f"\nTraining sample for curr_pos={curr_pos}:")
    print(f"Features: {features}")
    print(f"Block size: {training_sample['block_size']}")

    return training_sample

def augment_one_sample(model, tokenizer, device, prompt, model_args, gen_length=32, base_block_length=2, steps=16, correct_answer=None, break_after_answer_found=True):
    """
    Generate training samples by collecting confidence/entropy data at different curr_pos values,
    and save them to JSON and CSV files.
    
    Args:
        model: The model to use for generation
        tokenizer: Tokenizer for the model
        device: Device to run on
        prompt: Input prompt string
        model_args: Model arguments
        gen_length: Generation length (default: 32)
        base_block_length: Base block length (default: 2)
        steps: Number of steps (default: 16)
        correct_answer: Expected correct answer for checking correctness (optional)
        break_after_answer_found: If True, stop augmentation after first answer_found=True (default: True)
    
    Returns:
        List of training samples, each containing features and block_size
    """
    print(f"\n🔄 Augmenting sample with gen_length={gen_length}, base_block_length={base_block_length}, steps={steps}")
    
    # Get the token ID for <eot_id> to check for early termination
    eot_token_id = 126348  # Known <eot_id> token ID
    print(f"Using EOT token ID: {eot_token_id}")
    
    # Verify the token ID is correct
    try:
        token_text = tokenizer.decode([eot_token_id])
        print(f"Token ID {eot_token_id} decodes to: '{token_text}'")
    except Exception as e:
        print(f"Warning: Could not decode token ID {eot_token_id}: {e}")
    
    # Collect training samples
    training_samples = []
    manual_settings = {}
    
    for curr_pos in range(gen_length):
        print(f"\n=== curr_pos = {curr_pos} ===")
        if curr_pos > 0:  # empty for the first iteration
            manual_settings[curr_pos-1] = 1  # decode 1 token at a time for all previous positions
            print(f"manual_settings={manual_settings}")
        
        sample = generate_one_sample(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            model_args=model_args,
            gen_length=gen_length,
            base_block_length=base_block_length,
            steps=steps,
            curr_pos=curr_pos,
            manual_settings=manual_settings,
            correct_answer=correct_answer,
        )
        if sample:
            current_block_size = sample.get('block_size', 1)
            
            # Check if model has figured out the answer
            answer_found = (current_block_size == (gen_length - curr_pos))
            sample['answer_found'] = answer_found
            
            if answer_found:
                print(f"🎯 Answer found at curr_pos={curr_pos}!")
                print(f"   Block size {current_block_size} == remaining length {gen_length - curr_pos}")
                print(f"   Model wants to decode all remaining tokens at once")
                
                # Add this sample first, then decide whether to break
                training_samples.append(sample)
                
                if break_after_answer_found:
                    print(f"🛑 Breaking data augmentation (break_after_answer_found=True)")
                    break
                else:
                    print(f"➡️  Continuing data augmentation (break_after_answer_found=False)")
            else:
                training_samples.append(sample)
            
            # Check for <eot_id> token in the generated tokens
            if eot_token_id is not None and sample.get('full_token_ids'):
                full_token_ids = sample.get('full_token_ids', [])
                full_token_texts = sample.get('full_token_texts', [])
                
                # Check if we've generated an <eot_id> token at ANY position up to curr_pos
                for pos in range(min(curr_pos + 1, len(full_token_ids))):
                    if full_token_ids[pos] == eot_token_id:
                        token_text = full_token_texts[pos] if pos < len(full_token_texts) else "unknown"
                        print(f"🛑 Found <eot_id> token (ID: {eot_token_id}, text: '{token_text}') at position {pos}, stopping data augmentation")
                        print(f"   Generated tokens so far: {full_token_ids[:pos+1]}")
                        print(f"   Token texts: {full_token_texts[:pos+1]}")
                        break
                else:
                    # No <eot_id> found, continue
                    continue
                # If we found <eot_id>, break out of the main loop
                break
    
    # Save to JSON file
    import json
    json_output_path = "./data/sft_training_samples_greedy.json"
    with open(json_output_path, "w") as f:
        json.dump(training_samples, f, indent=2)
    
    # Save to CSV file for easier review
    import csv
    csv_output_path = "./data/sft_training_samples_greedy.csv"
    with open(csv_output_path, "w", newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = [
            'sample_id', 'confidence', 'entropy', 'position', 'token_id', 'token_text', 
            'position_relative', 'conf_0', 'entropy_0', 'top1_margin', 'mean_confidence', 'mean_entropy',
            'conf_std', 'entropy_std', 'conf_1', 'top4_conf_min', 'next4_conf_min',
            'top8_conf_min', 'next8_conf_min', 'block_size', 'answer_found',
            'full_confidence_list', 'full_entropy_list', 'full_token_ids', 'full_token_texts'
        ]
        writer.writerow(header)
        
        # Write data
        for sample_id, sample in enumerate(training_samples):
            block_size = sample['block_size']
            # Each sample now has exactly one feature row
            if sample['features']:  # Check if features exist
                feature = sample['features'][0]  # Get the single feature row
                if len(feature) >= 18:  # Full feature row (now has 18 features with position_relative)
                    (confidence, entropy, position, token_id, token_text, position_relative,
                     conf_0, entropy_0, top1_margin, mean_confidence, mean_entropy,
                     conf_std, entropy_std, conf_1, top4_conf_min, next4_conf_min,
                     top8_conf_min, next8_conf_min) = feature
                else:  # Fallback for incomplete rows
                    confidence, entropy, position = feature[:3]
                    token_id, token_text = feature[3:5] if len(feature) > 4 else (None, None)
                    position_relative = feature[5] if len(feature) > 5 else 0.0
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
                    sample.get('full_token_texts', [])
                ])
            else:
                # No features available, write a default row
                writer.writerow([sample_id, 0.0, 0.0, 0, None, None, 0.0] + [0.0] * 12 + [1, False, [], [], [], []])
    
    print(f"\n✅ Done. Saved {len(training_samples)} samples to:")
    print(f"  📄 JSON: {json_output_path}")
    print(f"  📊 CSV:  {csv_output_path}")
    
    return training_samples

def main(script_args, model_args, inference_args):
    """
    Args:
        script_args, training_args, model_args: Standard TRL arguments
        inference_args: InferenceArguments object containing inference-specific parameters
    """
    # # Convert string arguments to lists if they're strings
    # if isinstance(inference_args.bit_choices, str):
    #     bit_choices = [int(x.strip()) for x in inference_args.bit_choices.split(",")]
    # else:
    #     bit_choices = inference_args.bit_choices
        
    # if isinstance(inference_args.quant_layers, str):
    #     quant_layers = [int(x.strip()) for x in inference_args.quant_layers.split(",")]
    # else:
    #     quant_layers = inference_args.quant_layers
    
    # # Load validation examples from JSON
    # # with open(inference_args.eval_json_path, "r") as f:
    # #     dataset = [json.loads(line) for line in f][:inference_args.max_inf_size]
    # # print(f"Examples used for inference: {len(dataset)}")

    # ################
    # # Model & Tokenizer
    # ################
    # quantization_config = get_quantization_config(model_args)

    # # load tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path, 
    #     padding_side="left", 
    #     trust_remote_code=model_args.trust_remote_code,
    # )
    # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # if tokenizer.chat_template is None:
    #     tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # # load base model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path, 
    #     trust_remote_code=model_args.trust_remote_code,
    # ).to(device)
    # print(f"Loaded base model path: {model_args.model_name_or_path}")

    # ################
    # # Inference
    # ################

    # # Inference loop
    # predictions, references = [], []

    # print("\nINFERENCE:\n")

    # question = "What is 2*2?"
    # prompt = f"{question}\n"
    # print(f"prompt = \n{prompt}")

    # inputs = tokenizer(
    #     prompt,
    #     return_tensors="pt",
    #     padding=True,
    #     truncation=True,
    #     max_length=512,
    # ).to(device)

    # with torch.no_grad():
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=32,
    #         do_sample=False,
    #         use_cache=False, # required to disable KV cache
    #         eos_token_id=tokenizer.eos_token_id,
    #         pad_token_id=tokenizer.eos_token_id
    #     )

    # generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    # # generated_truncated = generated.split("\n")[0].strip()

    # print(f"generated=\n{generated}")

    return 0

def augment_multiple_samples(model, tokenizer, device, model_args, csv_path, num_questions=2, 
                           gen_length=32, base_block_length=1, steps=32, break_after_answer_found=True,
                           output_json_path="./data/sft_training_samples_multi_greedy.json",
                           output_csv_path="./data/sft_training_samples_multi_greedy.csv"):
    """
    Process multiple questions from a CSV file and generate training samples for each.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        device: The device (cuda/cpu)
        model_args: Model arguments
        csv_path: Path to CSV file containing questions
        num_questions: Number of questions to process (default: 2)
        gen_length: Generation length (default: 32)
        base_block_length: Base block length (default: 1)
        steps: Number of steps (default: 32)
        break_after_answer_found: Whether to break after answer found (default: True)
        output_json_path: Output JSON file path
        output_csv_path: Output CSV file path
    
    Returns:
        List of all training samples across all questions
    """
    import pandas as pd
    import json
    import csv
    
    # Load correct questions
    df = pd.read_csv(csv_path)
    
    # Truncate to desired number of questions
    df = df.head(num_questions)
    print(f"Processing {len(df)} questions (truncated from full dataset)")
    
    # Parameters
    instr = "Solve this problem and box your final answer:\n"
    
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
        
        # Generate training samples for this question
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
            break_after_answer_found=break_after_answer_found
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
    with open(output_json_path, "w") as f:
        json.dump(all_training_samples, f, indent=2)

    # Save to CSV file for easier review (same format as augment_one_sample)
    with open(output_csv_path, "w", newline='') as f:
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
    print(f"  📄 JSON: {output_json_path}")
    print(f"  📊 CSV:  {output_csv_path}")
    print(f"  📈 Samples per question: {len(all_training_samples) / len(df):.1f} avg")
    
    return all_training_samples

if __name__ == "__main__":
    # Parse HF script_args, model_args
    parser = make_parser()
    script_args, model_args = parser.parse_args_into_dataclasses()
    
    # Parse custom inference args
    args = parser.parse_args()
    bit_choices = [int(x.strip()) for x in args.bit_choices.split(",")] # convert to list
    quant_layers = [int(x.strip()) for x in args.quant_layers.split(",")] # convert to list
    inference_args = InferenceArguments(
        eval_json_path=args.eval_json_path,
        adapter_path=args.adapter_path,
        output_csv_path=args.output_csv_path,
        bitwise_lora_adapter_path=args.bitwise_lora_adapter_path,
        use_quantization=args.use_quantization,
        use_bitwise_lora=args.use_bitwise_lora,
        bit_choices=args.bit_choices,
        max_inf_size=args.max_inf_size,
        quant_layers=args.quant_layers,
        inf_bit_config=args.inf_bit_config,
        default_bit=args.default_bit
    )
    
    # Run inference
    main(script_args, model_args, inference_args)
