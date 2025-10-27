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
from generate import generate_vanilla, generate_custom, generate_charles

# FASTEST: Load model weights and recreate architecture
print("Loading saved model (fast method)...")

# Set memory management environment variables
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

if __name__ == '__main__':
    ########################################################
    # CONFIGURATION: Choose mode
    ########################################################
    USE_PARALLEL = False  # Set to False for sequential mode (needed for batch inference)
    NUM_GPUS = 1  # Only used if USE_PARALLEL=True
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

    # Recreate model architecture (fast) - use same method as original
    from transformers import AutoModel, AutoTokenizer

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
        # Set device - use the available GPU (H100 is on cuda:0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    # # Load gsm8k
    # df = load_gsm8k(start=0, end=1)

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
    
    # if USE_PARALLEL:
    #     print(f"🚀 Starting PARALLEL mode with {NUM_GPUS} GPUs...")
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
    #         num_gpus=NUM_GPUS
    #     )
    # else:
    #     print(f"🚀 Starting SEQUENTIAL mode...")
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
    #         instruction=instruction
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
    # Run inference with XGBoost scheduler (FAST ADAPTIVE INFERENCE)
    ########################################################
    # print("="*80)
    # print("🚀 XGBOOST SCHEDULER-GUIDED INFERENCE")
    # print("="*80)
    
    # # Load scheduler functions
    # from inference import load_scheduler, run_inference_batch_with_scheduler
    
    # # Configuration
    # SCHEDULER_PATH = "./cache/block_size_scheduler.json"  # Path to trained XGBoost model
    # USE_REGRESSION = True  # True for regression, False for classification
    # INPUT_CSV = "./data/gsm8k_correct.csv"  # Input questions
    # # INPUT_CSV = "./data/gsm8k.csv"  # Input questions
    # OUTPUT_CSV = "./output/predictions_with_scheduler.csv"  # Output predictions
    
    # # Generation settings
    # GEN_LENGTH = 32
    # BASE_BLOCK_LENGTH = 1
    # STEPS = 32
    
    # # Load scheduler
    # scheduler = load_scheduler(SCHEDULER_PATH, use_regression=USE_REGRESSION)
    
    # print(f"\n{'='*80}")
    # print("📊 INFERENCE SETTINGS")
    # print(f"{'='*80}")
    # print(f"  Input:  {INPUT_CSV}")
    # print(f"  Output: {OUTPUT_CSV}")
    # print(f"  Scheduler: {'Regression' if USE_REGRESSION else 'Classification'}")
    # print(f"  Generation: gen_length={GEN_LENGTH}, steps={STEPS}")
    # print(f"{'='*80}\n")
    
    # # Run inference
    # start_time = time.time()
    # results_df = run_inference_batch_with_scheduler(
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=device,
    #     scheduler=scheduler,
    #     model_args=model_args,
    #     input_csv_path=INPUT_CSV,
    #     output_csv_path=OUTPUT_CSV,
    #     steps=STEPS,
    #     gen_length=GEN_LENGTH,
    #     base_block_length=BASE_BLOCK_LENGTH,
    #     use_regression=USE_REGRESSION,
    #     instruction=instruction
    # )
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    
    # print(f"\n{'='*80}")
    # print("✅ INFERENCE COMPLETE!")
    # print(f"{'='*80}")
    # print(f"\n⏱️  TIMING REPORT:")
    # print(f"  📊 Questions processed: {len(results_df)}")
    # print(f"  ⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    # print(f"  ⚡ Time per question: {elapsed_time/len(results_df):.2f} seconds")

    ########################################################
    # Run inference with XGBoost scheduler (CHARLES)
    ########################################################
    print("="*80)
    print("🚀 CHARLES: XGBoost-Guided Dynamic Block Size Inference")
    print("="*80)
    
    # Load XGBoost scheduler
    import xgboost as xgb
    SCHEDULER_PATH = "./cache/block_size_scheduler.json"  # Path to trained XGBoost model
    USE_REGRESSION = True  # True for regression, False for classification
    
    print(f"📥 Loading XGBoost scheduler from: {SCHEDULER_PATH}")
    scheduler = xgb.XGBRegressor() if USE_REGRESSION else xgb.XGBClassifier()
    scheduler.load_model(SCHEDULER_PATH)
    print(f"✅ Scheduler loaded ({'Regression' if USE_REGRESSION else 'Classification'} model)")
    
    # Generation settings
    GEN_LENGTH = 32
    STEPS = 32  # Not used in dynamic version, kept for compatibility
    TEMPERATURE = 0.
    CFG_SCALE = 0.
    REMASKING = 'low_confidence'
    
    print(f"\n{'='*80}")
    print("📊 GENERATION SETTINGS")
    print(f"{'='*80}")
    print(f"  Generation length: {GEN_LENGTH}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  CFG scale: {CFG_SCALE}")
    print(f"  Remasking: {REMASKING}")
    print(f"  Expected answer: {correct_answer}")
    print(f"{'='*80}\n")
    
    print(f"📝 Question: {prompt}\n")
    
    # Tokenize prompt
    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    # Run generation with XGBoost scheduler
    print("🎯 Starting dynamic block size generation...")
    start_time = time.time()
    
    out = generate_charles(
        model=model,
        tokenizer=tokenizer,
        prompt=input_ids,
        scheduler=scheduler,
        steps=STEPS,
        gen_length=GEN_LENGTH,
        block_length=1,  # Fallback if scheduler is None
        temperature=TEMPERATURE,
        cfg_scale=CFG_SCALE,
        remasking=REMASKING,
        expected_answer=correct_answer,
        use_regression=USE_REGRESSION
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Decode output
    generated_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    print(f"\n{'='*80}")
    print("✅ GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📄 Generated Answer:\n{generated_text}")
    
    # Extract numerical answer
    from generate import extract_numerical
    predicted_answer = extract_numerical(generated_text)
    is_correct = (predicted_answer == correct_answer) if predicted_answer is not None else False
    
    print(f"\n{'='*80}")
    print("📊 RESULTS")
    print(f"{'='*80}")
    print(f"  Expected answer: {correct_answer}")
    print(f"  Predicted answer: {predicted_answer}")
    print(f"  Correct: {'✅ YES' if is_correct else '❌ NO'}")
    print(f"\n⏱️  Generation time: {elapsed_time:.2f} seconds")
    print(f"  Tokens generated: {GEN_LENGTH}")
    print(f"  Tokens per second: {GEN_LENGTH/elapsed_time:.2f}")
    print(f"{'='*80}\n")

