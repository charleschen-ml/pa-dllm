# -*- coding: utf-8 -*-
"""
Run Inference Script - Load saved model and run inference
"""

# Set HuggingFace cache directories to local project cache
import os

from numpy import true_divide
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
from inference import augment_one_sample, augment_one_sample_greedy, augment_one_sample_dispatch, load_gsm8k, augment_multiple_samples
from generate import generate_vanilla, generate_custom, generate_charles

# FASTEST: Load model weights and recreate architecture
print("Loading saved model (fast method)...")

# Set memory management environment variables
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ==============================================================================
# MODEL LOADING FUNCTIONS
# ==============================================================================

def load_model_hf(model_args, device):
    """
    Load HuggingFace model from cached weights.
    
    This loads the model architecture from HF and then loads the cached weights
    for faster loading without re-downloading.
    
    Args:
        model_args: ModelConfig object with model_name_or_path, trust_remote_code, etc.
        device: torch.device to load model onto
    
    Returns:
        (model, tokenizer): Loaded model and tokenizer
    """
    from transformers import AutoModel, AutoTokenizer
    
    # Load model architecture first (empty model)
    print("Loading model architecture...")
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    print("✅ Model architecture loaded")

    # Load saved weights (much faster than downloading)
    print("Loading saved model weights...")
    state_dict = torch.load('./cache/model_weights.pt', weights_only=True, map_location='cpu')
    model.load_state_dict(state_dict)

    # Now move to GPU (removed device_map="auto" to avoid multi-GPU distribution)
    model = model.to(device).eval()
    print("✅ Model loaded from saved weights (fast)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./cache/tokenizer/')

    print(f"✅ Loaded HF model: {model_args.model_name_or_path} on {device}")
    print(f"📁 Cache location: ./cache/")

    # Memory usage info
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
    
    return model, tokenizer


def load_model_custom(model_args, device):
    """
    Load custom model from local modeling_llada.py.
    
    This allows you to modify the model architecture in your local modeling_llada.py
    while still loading the pretrained weights from cache.
    
    Args:
        model_args: ModelConfig object with model_name_or_path, trust_remote_code, etc.
        device: torch.device to load model onto
    
    Returns:
        (model, tokenizer): Loaded model and tokenizer
    """
    from transformers import AutoTokenizer
    from modeling_llada import LLaDAModelLM, LLaDAConfig
    
    # Load config from pretrained (can use cache or HF)
    print("Loading config from pretrained...")
    config = LLaDAConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    print("✅ Config loaded")
    
    # Enable scheduler head for block_size_rel prediction
    config.use_scheduler_head = True
    print("✅ Scheduler head enabled")
    
    # Create model using LOCAL modeling_llada.py
    print("Creating model from LOCAL modeling_llada.py...")
    model = LLaDAModelLM(config, init_params=False)
    print("✅ Model architecture created from local code")
    
    # Debug: Check if scheduler_head was created
    if hasattr(model.model.transformer, "scheduler_head"):
        print("✅ Scheduler head found in model.model.transformer")
    else:
        print("❌ Scheduler head NOT found in model.model.transformer")
        print(f"   transformer keys: {list(model.model.transformer.keys())[:10]}...")
    
    # Load saved weights (much faster than downloading)
    print("Loading saved model weights...")
    state_dict = torch.load('./cache/model_weights.pt', weights_only=True, map_location='cpu')
    
    # Load weights with strict=False to allow new scheduler_head (not in saved weights)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️  Missing keys (newly added components): {missing_keys}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {unexpected_keys}")
    
    # Now move to GPU and set to eval mode
    model = model.to(device).to(torch.bfloat16).eval()
    print("✅ Model loaded from saved weights (fast)")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./cache/tokenizer/')
    
    print(f"✅ Loaded CUSTOM model from local modeling_llada.py on {device}")
    print(f"📁 Cache location: ./cache/")
    
    # Memory usage info
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.1f}GB, Reserved: {reserved:.1f}GB")
    
    return model, tokenizer


def load_model(model_args, use_custom=False):
    """
    Dispatcher function to load model (HF or custom).
    
    Args:
        model_args: ModelConfig object with model_name_or_path, trust_remote_code, etc.
        use_custom: If True, use custom local modeling_llada.py; if False, use HF model
    
    Returns:
        (model, tokenizer, device): Loaded model, tokenizer, and device
    """
    # Set device - use the available GPU (H100 is on cuda:0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if use_custom:
        model, tokenizer = load_model_custom(model_args, device)
    else:
        model, tokenizer = load_model_hf(model_args, device)
    
    return model, tokenizer, device


if __name__ == '__main__':
    ########################################################
    # CONFIGURATION: Choose mode
    ########################################################
    USE_CUSTOM_MODEL = True  # True: use custom local modeling_llada.py, False: use HF model
    USE_GREEDY = True  # True: use greedy mode when generating training samples, False: use AR mode
    USE_PARALLEL = False  # Set to False for sequential mode (needed for batch inference)
    NUM_GPUS = 4  # Only used if USE_PARALLEL=True
    NUM_QUESTIONS = 1  # Number of questions to process (None = process all questions in CSV)
    
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

    # Clear any existing GPU memory first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Load model only for sequential mode
    if USE_PARALLEL:
        print("⚠️  Skipping model load in main process (parallel mode)")
        print(f"🚀 Workers will load models on {NUM_GPUS} GPUs")
        model = None
        tokenizer = None
        device = None
    else:
        print("📦 Loading model for sequential mode...")
        model, tokenizer, device = load_model(model_args, use_custom=USE_CUSTOM_MODEL)

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
    # # Load gsm8k
    # df = load_gsm8k(start=0, end=2000)

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
    #     block_length=1,
    #     instruction=instruction
    # )
    # # Load df from csv
    # df = pd.read_csv("./data/gsm8k_output.csv")
    # # Calculate score
    # correct_path = "./data/gsm8k_correct.csv"
    # calculate_score(df, correct_path)

    ########################################################
    # Load single prompt (only for sequential mode)
    ########################################################
    if not USE_PARALLEL:
        df = pd.read_csv("./data/gsm8k_correct.csv")
        question = df.iloc[0]['question']
        correct_answer = int(df.iloc[0]['answer_numerical'])
        if instruction is not None:
            question = instruction + question
        prompt = question

    ########################################################
    # Run single inference
    ########################################################
    run_inference(model, tokenizer, device, prompt, model_args, gen_length=32, base_block_length=1, steps=32)

    ########################################################
    # Run greedy inference
    ########################################################
    # run_greedy_inference(model, tokenizer, device, prompt, model_args, gen_length=32, base_block_length=1, steps=32)

    ########################################################
    # Generate one sample
    ########################################################
    # print("🚀 Starting generate_one_sample...")
    # start_time = time.time()
    # manual_settings = {}
    # training_sample = generate_one_sample(
    #     model, tokenizer, device, prompt, model_args, 
    #     gen_length=32, 
    #     base_block_length=1, 
    #     steps=32, 
    #     curr_pos=0, 
    #     correct_answer=correct_answer,
    #     manual_settings=manual_settings,)
    # print(f"training_sample=\n{training_sample}")
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"\n⏱️  TIMING REPORT:")
    # print(f"  ⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")

    ########################################################
    # Augment one sample
    ########################################################
    # mode_str = "GREEDY" if USE_GREEDY else "AR"
    # print(f"🚀 Starting augment_one_sample ({mode_str} mode)...")
    # start_time = time.time()
    
    # training_samples = augment_one_sample_dispatch(
    #     use_greedy=USE_GREEDY,
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
    # Augment multiple samples: Sequential or Parallel
    ########################################################
    # start_time = time.time()
    
    # # Determine how many questions to process
    # csv_path = "./data/gsm8k_correct.csv"
    # df_temp = pd.read_csv(csv_path)
    # total_questions = len(df_temp)
    # print(f"📊 Found {total_questions} questions in {csv_path}")
    
    # # Use NUM_QUESTIONS if specified, otherwise process all
    # questions_to_process = NUM_QUESTIONS if NUM_QUESTIONS is not None else total_questions
    # print(f"🎯 Processing {questions_to_process} questions")
    
    # mode_str = "GREEDY" if USE_GREEDY else "AR"
    # if USE_PARALLEL:
    #     print(f"🚀 Starting PARALLEL mode with {NUM_GPUS} GPUs ({mode_str} mode)...")
    #     from inference import augment_multiple_samples_parallel
        
    #     all_training_samples = augment_multiple_samples_parallel(
    #         model_args=model_args,
    #         csv_path=csv_path,
    #         num_questions=questions_to_process,
    #         gen_length=32,
    #         base_block_length=1,
    #         steps=32,
    #         break_after_answer_found=True,
    #         output_json_path="./data/sft_training_samples_multi_greedy_parallel.json",
    #         output_csv_path="./data/sft_training_samples_multi_greedy_parallel.csv",
    #         instruction=instruction,
    #         num_gpus=NUM_GPUS,
    #         use_greedy=USE_GREEDY
    #     )
    # else:
    #     print(f"🚀 Starting SEQUENTIAL mode ({mode_str} mode)...")
    #     all_training_samples = augment_multiple_samples(
    #         model=model,
    #         tokenizer=tokenizer,
    #         device=device,
    #         model_args=model_args,
    #         csv_path=csv_path,
    #         num_questions=questions_to_process,
    #         gen_length=32,
    #         base_block_length=1,
    #         steps=32,
    #         break_after_answer_found=True,
    #         output_json_path="./data/sft_training_samples_multi_greedy.json",
    #         output_csv_path="./data/sft_training_samples_multi_greedy.csv",
    #         instruction=instruction,
    #         use_greedy=USE_GREEDY
    #     )
    
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"\n⏱️  TIMING REPORT:")
    # print(f"  📊 Total samples generated: {len(all_training_samples)}")
    # print(f"  ⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    # print(f"  ⚡ Time per sample: {elapsed_time/len(all_training_samples):.2f} seconds")
    # if USE_PARALLEL:
    #     print(f"  🚀 Used {NUM_GPUS} GPUs in parallel!")
    #     print(f"  🎯 Processing rate: {len(all_training_samples)/elapsed_time:.1f} samples/second")

    ########################################################
    # Run inference with XGBoost scheduler (CHARLES)
    ########################################################
    # print("="*80)
    # print("🚀 CHARLES: XGBoost-Guided Dynamic Block Size Inference")
    # print("="*80)
    
    # # Load XGBoost scheduler
    # import xgboost as xgb
    # SCHEDULER_PATH = "./cache/block_size_scheduler.json"  # Path to trained XGBoost model
    # USE_REGRESSION = True  # True for regression, False for classification
    
    # print(f"📥 Loading XGBoost scheduler from: {SCHEDULER_PATH}")
    # scheduler = xgb.XGBRegressor() if USE_REGRESSION else xgb.XGBClassifier()
    # scheduler.load_model(SCHEDULER_PATH)
    # print(f"✅ Scheduler loaded ({'Regression' if USE_REGRESSION else 'Classification'} model)")
    
    # # Generation settings
    # GEN_LENGTH = 32
    # STEPS = 32  # Not used in dynamic version, kept for compatibility
    # TEMPERATURE = 0.
    # CFG_SCALE = 0.
    # REMASKING = 'low_confidence'
    # BLOCK_SIZE_OFFSET = 0  # Conservative offset: subtract from predicted block_size (0=no offset)
    # MAX_BLOCK_SIZE = 10  # Maximum allowed block size (should match training data cap)
    
    # # Load all questions from gsm8k_correct.csv
    # df_questions = pd.read_csv("./data/gsm8k_correct.csv")
    
    # # Limit to NUM_QUESTIONS if specified
    # if NUM_QUESTIONS is not None:
    #     df_questions = df_questions.head(NUM_QUESTIONS)
    
    # print(f"\n{'='*80}")
    # print("📊 GENERATION SETTINGS")
    # print(f"{'='*80}")
    # print(f"  Generation length: {GEN_LENGTH}")
    # print(f"  Temperature: {TEMPERATURE}")
    # print(f"  CFG scale: {CFG_SCALE}")
    # print(f"  Remasking: {REMASKING}")
    # print(f"  Block size offset: {BLOCK_SIZE_OFFSET} (conservative adjustment)")
    # print(f"  Max block size: {MAX_BLOCK_SIZE} (matches training data cap)")
    # print(f"  Total questions: {len(df_questions)}")
    # print(f"{'='*80}\n")
    
    # # Store results
    # results = []
    # total_start_time = time.time()
    
    # # Loop through all questions
    # for idx, row in df_questions.iterrows():
    #     question = row['question']
    #     correct_answer = int(row['answer_numerical'])
        
    #     if instruction is not None:
    #         question = instruction + question
    #     prompt = question
        
    #     print(f"\n{'='*80}")
    #     print(f"📝 Question {idx+1}/{len(df_questions)}")
    #     print(f"{'='*80}")
    #     print(f"Question: {prompt}")
    #     print(f"Expected answer: {correct_answer}\n")

    #     # Add special tokens for the Instruct model (not required for base model)
    #     m = [{"role": "user", "content": prompt}, ]
    #     prompt_formatted = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    #     input_ids = tokenizer(prompt_formatted)['input_ids']
    #     input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
        
    #     # Run generation with XGBoost scheduler
    #     print("🎯 Starting dynamic block size generation (CHARLES)...")
    #     start_time = time.time()
        
    #     out, num_steps = generate_charles(
    #         model=model,
    #         tokenizer=tokenizer,
    #         prompt=input_ids,
    #         scheduler=scheduler,
    #         steps=STEPS,
    #         gen_length=GEN_LENGTH,
    #         block_length=1,  # Fallback if scheduler is None
    #         temperature=TEMPERATURE,
    #         cfg_scale=CFG_SCALE,
    #         remasking=REMASKING,
    #         expected_answer=correct_answer,
    #         use_regression=USE_REGRESSION,
    #         block_size_offset=BLOCK_SIZE_OFFSET,
    #         max_block_size=MAX_BLOCK_SIZE
    #     )
        
    #     end_time = time.time()
    #     generation_time = end_time - start_time
        
    #     # Calculate speedup vs autoregressive (AR takes gen_length steps)
    #     ar_steps = GEN_LENGTH  # Autoregressive = 1 token per step = gen_length steps
    #     speedup = ar_steps / num_steps if num_steps > 0 else 0
        
    #     # Decode output
    #     generated_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        
    #     print(f"\n{'='*80}")
    #     print("✅ GENERATION COMPLETE!")
    #     print(f"{'='*80}")
    #     print(f"\n📄 Generated Answer:\n{generated_text}")
        
    #     # Extract numerical answer
    #     from generate import extract_numerical
    #     predicted_answer = extract_numerical(generated_text)
    #     is_correct = (predicted_answer == correct_answer) if predicted_answer is not None else False
        
    #     print(f"\n{'='*80}")
    #     print("📊 RESULTS")
    #     print(f"{'='*80}")
    #     print(f"  Expected answer: {correct_answer}")
    #     print(f"  Predicted answer: {predicted_answer}")
    #     print(f"  Correct: {'✅ YES' if is_correct else '❌ NO'}")
    #     print(f"\n⚡ EFFICIENCY:")
    #     print(f"  AR steps (baseline): {ar_steps}")
    #     print(f"  CHARLES steps: {num_steps}")
    #     print(f"  Speedup: {speedup:.2f}x")
    #     print(f"\n⏱️  Generation time: {generation_time:.2f} seconds")
    #     print(f"  Tokens per second: {GEN_LENGTH/generation_time:.2f}")
    #     print(f"{'='*80}\n")
        
    #     # Store results
    #     results.append({
    #         'question': row['question'],
    #         'correct_answer': correct_answer,
    #         'predicted_answer': predicted_answer,
    #         'generated_text': generated_text,
    #         'is_correct': is_correct,
    #         'ar_steps': ar_steps,
    #         'num_steps': num_steps,
    #         'speedup': speedup,
    #         'generation_time': generation_time
    #     })
    
    # # Calculate overall statistics
    # total_end_time = time.time()
    # total_elapsed_time = total_end_time - total_start_time
    
    # results_df = pd.DataFrame(results)
    # num_correct = results_df['is_correct'].sum()
    # accuracy = num_correct / len(results_df) * 100
    
    # # Calculate average speedup for correct answers only
    # correct_df = results_df[results_df['is_correct'] == True]
    # if len(correct_df) > 0:
    #     avg_speedup_correct = correct_df['speedup'].mean()
    #     avg_ar_steps = correct_df['ar_steps'].mean()
    #     avg_num_steps = correct_df['num_steps'].mean()
    #     avg_generation_time = correct_df['generation_time'].mean()
    # else:
    #     avg_speedup_correct = 0
    #     avg_ar_steps = 0
    #     avg_num_steps = 0
    #     avg_generation_time = 0
    
    # print(f"\n{'='*80}")
    # print("📊 OVERALL RESULTS")
    # print(f"{'='*80}")
    # print(f"  Total questions: {len(results_df)}")
    # print(f"  Correct answers: {num_correct}/{len(results_df)}")
    # print(f"  Accuracy: {accuracy:.2f}%")
    # print(f"\n⚡ EFFICIENCY (for correct answers only):")
    # print(f"  Average AR steps (baseline): {avg_ar_steps:.1f}")
    # print(f"  Average CHARLES steps: {avg_num_steps:.1f}")
    # print(f"  Average speedup: {avg_speedup_correct:.2f}x")
    # print(f"\n⏱️  TIMING:")
    # print(f"  Average generation time: {avg_generation_time:.2f} seconds")
    # print(f"  Total wall time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.1f} minutes)")
    # print(f"  Average wall time per question: {total_elapsed_time/len(results_df):.2f} seconds")
    # print(f"{'='*80}\n")
    
    # # Save results to CSV
    # output_path = "./output/charles_inference_results.csv"
    # results_df.to_csv(output_path, index=False)
    # print(f"💾 Results saved to: {output_path}")