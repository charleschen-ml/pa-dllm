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
from inference import run_inference_with_scheduler, write_to_json_csv, generate_interpolations
from generate import generate_vanilla, generate_custom, generate_charles
from calculate_interpolations import calculate_interpolations

# ==============================================================================
# MLP SCHEDULER MODEL DEFINITION
# ==============================================================================
import torch.nn as nn

class FeatureOnlySchedulerMLP(nn.Module):
    """
    Simple MLP that takes XGBoost features (+ optionally hidden states) and predicts block_size.
    
    Supports both classification (4 classes: 1, 2, 3, 4 tokens) and regression (block_size_rel).
    """
    
    def __init__(self, input_dim=30, hidden_dims=[256, 128, 64], num_classes=None):
        """
        Args:
            input_dim: Number of input features (30 for XGBoost features, or 30+4096 for features+hidden)
            hidden_dims: List of hidden layer dimensions
            num_classes: If provided, do classification with num_classes outputs.
                        If None, do regression with 1 output in [0, 1].
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Regularization
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Output layer
        if num_classes is not None:
            # Classification: output logits for each class
            self.output_layer = nn.Linear(prev_dim, num_classes)
        else:
            # Regression: output single value in [0, 1]
            self.output_layer = nn.Sequential(
                nn.Linear(prev_dim, 1),
                nn.Sigmoid()
            )
        
        # Store device for convenience
        self.device = None
    
    def forward(self, features):
        """
        Args:
            features: [batch_size, input_dim] tensor of features
        
        Returns:
            If classification: [batch_size, num_classes] logits
            If regression: [batch_size] predictions in [0, 1]
        """
        out = self.backbone(features)  # [batch_size, hidden_dim]
        out = self.output_layer(out)  # [batch_size, num_classes] or [batch_size, 1]
        
        if self.num_classes is None:
            # Regression: squeeze to [batch_size]
            return out.squeeze(-1)
        else:
            # Classification: return logits [batch_size, num_classes]
            return out

# ==============================================================================
# MODEL LOADING
# ==============================================================================

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
    
    # Set init device to CPU (will move to GPU later)
    config.init_device = "cpu"
    
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
        # Initialize scheduler_head if it was missing from state_dict
        if any('scheduler_head' in key for key in missing_keys):
            print("🔧 Initializing scheduler_head parameters...")
            for module in model.model.transformer.scheduler_head:
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
            print("✅ Scheduler_head initialized")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys: {unexpected_keys}")
    
    # Now move to GPU and convert to bfloat16
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
    USE_CUSTOM_MODEL = None  # True: use custom local modeling_llada.py, False: use HF model
                               # NOTE: Will be automatically set to True if SCHEDULER_TYPE='neural'
    USE_GREEDY = True  # True: use greedy mode when generating training samples, False: use AR mode
    USE_PARALLEL = False  # Set to False for sequential mode (needed for batch inference)
    NUM_GPUS = 2  # Only used if USE_PARALLEL=True
    NUM_QUESTIONS = None  # Number of questions to process (None = process all questions in CSV)
    BLOCK_SIZE_MAX = 4  # Maximum block size to cap sweep values at (None = no cap)
    EVALUATION_SPLIT = 'test'  # Which split to evaluate on: None, 'train', 'val', 'test'
                             # Set to 'train'/'val'/'test' to run inference only on that split
    BASELINE_STRATEGY = None  # Baseline strategy for testing: None, 'always_1', 'always_2', 'always_4', 'random_uniform'
                              # None: Use XGBoost scheduler (normal mode)
                              # 'random_uniform': Splits equally across 1 to block_size_max (e.g., 25-25-25-25 for max=4)
                              # 'always_1': Always predict block_size=1 (conservative, ~100% acc, 1.0x speedup)
                              # 'always_2': Always predict block_size=2 (aggressive, higher speedup)
                              # 'always_4': Always predict block_size=4 (very aggressive, upper bound speedup for 4tok)
    
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
        
        # If using neural scheduler, load the trained scheduler head weights
        if USE_CUSTOM_MODEL and hasattr(model, 'model') and hasattr(model.model.transformer, 'scheduler_head'):
            from inf_scheduler_head import load_trained_scheduler_head
            SCHEDULER_HEAD_CHECKPOINT = 'checkpoints/scheduler_head/scheduler_head_best.pt'
            print(f"\n🧠 Loading trained scheduler head weights from {SCHEDULER_HEAD_CHECKPOINT}...")
            model = load_trained_scheduler_head(model, SCHEDULER_HEAD_CHECKPOINT)

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
    # df = load_gsm8k(start=5000, end=8000)

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
    # run_greedy_inference(model, tokenizer, device, prompt, model_args, gen_length=32, base_block_length=1, steps=32)

    ########################################################
    # Generate one sample
    ########################################################
    # print("🚀 Starting generate_one_sample...")
    # start_time = time.time()
    # manual_settings = {0:1}
    # curr_pos = max(manual_settings.keys()) + 1
    # training_sample = generate_one_sample(
    #     model, tokenizer, device, prompt, model_args, 
    #     gen_length=32, 
    #     base_block_length=1, 
    #     steps=32, 
    #     curr_pos=curr_pos, # block index to optimize
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
    mode_str = "GREEDY" if USE_GREEDY else "AR"
    print(f"🚀 Starting augment_one_sample ({mode_str} mode)...")
    start_time = time.time()
    
    training_samples = augment_one_sample_dispatch(
        use_greedy=USE_GREEDY,
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=prompt,
        model_args=model_args,
        gen_length=32,
        base_block_length=1,
        steps=32,
        correct_answer=correct_answer,
        break_after_answer_found=True,  # Set to False to continue augmentation after answer found
        block_size_max=BLOCK_SIZE_MAX
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n⏱️  TIMING REPORT:")
    print(f"  ⏱️  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")

    ########################################################
    # Augment multiple samples: Sequential or Parallel
    #######################################################
    # start_time = time.time()
    
    # # Determine how many questions to process
    # csv_path = "./data/gsm8k_correct_5k_to_8k.csv"
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
    #         output_json_path="./data/sft_training_samples_multi_greedy_parallel_4tok_5k_to_8k.json",
    #         output_csv_path="./data/sft_training_samples_multi_greedy_parallel_4tok_5k_to_8k.csv",
    #         instruction=instruction,
    #         num_gpus=NUM_GPUS,
    #         use_greedy=USE_GREEDY,
    #         block_size_max=BLOCK_SIZE_MAX
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
    #         use_greedy=USE_GREEDY,
    #         block_size_max=BLOCK_SIZE_MAX
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
    # Generate interpolations from greedy samples
    ########################################################
    # interpolated_samples = generate_interpolations(
    #     model=model,
    #     tokenizer=tokenizer,
    #     device=device,
    #     model_args=model_args,
    #     greedy_csv_path="./data/sft_training_samples_multi_greedy_parallel.csv",
    #     output_csv_path="./data/sft_training_samples_interpolated.csv",
    #     num_questions=NUM_QUESTIONS,
    #     instruction=None,  # prompt already has instruction in the CSV
    #     block_size_max=BLOCK_SIZE_MAX
    # )

    ########################################################
    # Run inference with scheduler (CHARLES)
    ########################################################
    # ============================================================================
    # MLP SCHEDULER CONFIGURATION - Easy Mode Switching
    # ============================================================================
    # Choose one of three modes:
    #   'dual_stream'   - Features + Hidden States with fusion (73.9% test acc)
    #   'hidden_only'   - Hidden States only (72.6% test acc) 
    #   'features_only' - XGBoost Features only (65.4% test acc)
    # ============================================================================
    MLP_MODE = 'hidden_only'  # ⭐ CHANGE THIS TO SWITCH MODES
    # ============================================================================
    
    # Configuration
    SCHEDULER_TYPE = 'mlp_features'  # 'xgboost', 'neural', or 'mlp_features'
    
    # Auto-enable custom model if using neural scheduler
    if SCHEDULER_TYPE == 'neural' and not USE_CUSTOM_MODEL:
        print("\n" + "="*80)
        print("⚠️  CRITICAL: Neural scheduler requires custom model!")
        print("="*80)
        print("The HuggingFace model doesn't have scheduler head modifications.")
        print("Please set USE_CUSTOM_MODEL = True at the top of the script")
        print("and re-run to load the model with scheduler_head enabled.")
        print("="*80 + "\n")
        raise ValueError("USE_CUSTOM_MODEL must be True when SCHEDULER_TYPE='neural'")
    
    # Load external MLP scheduler if using mlp_features
    mlp_scheduler = None
    if SCHEDULER_TYPE == 'mlp_features':
        # Configure based on selected mode
        if MLP_MODE == 'dual_stream':
            MLP_CHECKPOINT_PATH = "checkpoints/scheduler_head_dual_stream/mlp_best.pt"
            USE_DUAL_STREAM_INFERENCE = True
            MODE_DESCRIPTION = "Dual-Stream (Features + Hidden States with Fusion)"
        elif MLP_MODE == 'hidden_only':
            MLP_CHECKPOINT_PATH = "checkpoints/ablation_hidden_only/mlp_best.pt"
            INPUT_DIM = 4096  # Hidden states only
            USE_DUAL_STREAM_INFERENCE = False
            MODE_DESCRIPTION = "Hidden States Only"
        elif MLP_MODE == 'features_only':
            MLP_CHECKPOINT_PATH = "checkpoints/ablation_features_only/mlp_best.pt"
            INPUT_DIM = 30  # Features only
            USE_DUAL_STREAM_INFERENCE = False
            MODE_DESCRIPTION = "XGBoost Features Only"
        else:
            raise ValueError(f"Invalid MLP_MODE: {MLP_MODE}. Choose 'dual_stream', 'hidden_only', or 'features_only'")
        
        print("\n" + "="*80)
        print(f"📥 Loading External MLP Scheduler: {MODE_DESCRIPTION}")
        print("="*80)
        print(f"   Mode: {MLP_MODE}")
        print(f"   Checkpoint: {MLP_CHECKPOINT_PATH}")
        
        if USE_DUAL_STREAM_INFERENCE:
            # Dual-stream model - need to import DualStreamSchedulerMLP
            from train_scheduler_head_features import DualStreamSchedulerMLP
            HIDDEN_DIMS = [256, 128]
            NUM_CLASSES = 4
            
            mlp_scheduler = DualStreamSchedulerMLP(
                xgb_feature_dim=30,
                hidden_size=4096,
                projection_dim=512,
                mlp_hidden_dims=HIDDEN_DIMS,
                num_classes=NUM_CLASSES
            ).to(device).eval()
            
            print(f"   Architecture: Dual-Stream MLP")
            print(f"     Stream 1 (Hidden): [seq_len, 4096] → MLP → pool → [4]")
            print(f"     Stream 2 (Features): [30] → MLP → [4]")
            print(f"     Fusion: Per-class weighted sum")
        else:
            # Single-stream model
            HIDDEN_DIMS = [256, 128]
            NUM_CLASSES = 4
            
            mlp_scheduler = FeatureOnlySchedulerMLP(
                input_dim=INPUT_DIM,
                hidden_dims=HIDDEN_DIMS,
                num_classes=NUM_CLASSES
            ).to(device).eval()
            
            print(f"   Input dim: {INPUT_DIM}")
            print(f"   Architecture: {INPUT_DIM} → {' → '.join(map(str, HIDDEN_DIMS))} → {NUM_CLASSES}")
        
        # Load checkpoint
        checkpoint = torch.load(MLP_CHECKPOINT_PATH, map_location=device)
        mlp_scheduler.load_state_dict(checkpoint['model_state_dict'])
        mlp_scheduler.device = device  # Store device for convenience
        mlp_scheduler.mlp_mode = MLP_MODE  # Store mode for generate.py to use
        mlp_scheduler.use_dual_stream = USE_DUAL_STREAM_INFERENCE  # Store for generate.py
        
        print(f"✅ MLP Scheduler loaded successfully!")
        print(f"   Trained for {checkpoint['epoch']} epochs")
        print(f"   Best validation loss: {checkpoint.get('val_loss', 'N/A')}")
        print("="*80 + "\n")
    
    SCHEDULER_PATH = "./cache/block_size_scheduler.json"  # Path to trained XGBoost model (only for 'xgboost')
    USE_REGRESSION = False  # True for regression, False for classification (only for 'xgboost')
    GEN_LENGTH = 128
    STEPS = 32  # Not used in dynamic version, kept for compatibility
    TEMPERATURE = 0.
    CFG_SCALE = 0.
    REMASKING = 'low_confidence'
    BLOCK_SIZE_OFFSET = 0  # Conservative offset: subtract from predicted block_size (0=no offset)
    MAX_BLOCK_SIZE = 4  # Maximum allowed block size (should match training data cap)
    
    # Run scheduler inference
    output_suffix = f"_{EVALUATION_SPLIT}" if EVALUATION_SPLIT else ""
    results_df = run_inference_with_scheduler(
        model=model,
        tokenizer=tokenizer,
        device=device,
        scheduler_path=SCHEDULER_PATH,
        questions_csv_path="./data/gsm8k_correct.csv",
        num_questions=NUM_QUESTIONS,
        gen_length=GEN_LENGTH,
        steps=STEPS,
        temperature=TEMPERATURE,
        cfg_scale=CFG_SCALE,
        remasking=REMASKING,
        block_size_offset=BLOCK_SIZE_OFFSET,
        max_block_size=MAX_BLOCK_SIZE,
        use_regression=USE_REGRESSION,
        instruction=instruction,
        output_path=f"./output/charles_inference_results{output_suffix}.csv",
        scheduler_type=SCHEDULER_TYPE,
        evaluation_split=EVALUATION_SPLIT,
        baseline_strategy=BASELINE_STRATEGY,
        mlp_scheduler=mlp_scheduler
    )