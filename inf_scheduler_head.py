"""
Inference script to test the trained scheduler head.

This script:
1. Loads the base LLaDA model with scheduler head
2. Loads trained scheduler head weights from checkpoint
3. Runs inference on test samples
4. Compares predicted vs actual block_size_rel
"""

import json
import torch
from run_inference import load_model


def load_trained_scheduler_head(model, checkpoint_path):
    """
    Load trained scheduler head weights into the model.
    
    Args:
        model: LLaDAModelLM with scheduler_head
        checkpoint_path: Path to checkpoint file
    
    Returns:
        model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load only the scheduler head weights
    scheduler_head_state = checkpoint['scheduler_head_state_dict']
    model.model.transformer.scheduler_head.load_state_dict(scheduler_head_state)
    
    print(f"✅ Loaded scheduler head from: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Train Loss: {checkpoint['train_loss']:.6f}")
    print(f"   Val Loss: {checkpoint['val_loss']:.6f}")
    
    return model


def run_inference_on_sample(model, tokenizer, sample, device):
    """
    Run inference on a single sample and return predictions.
    
    Args:
        model: LLaDAModelLM with trained scheduler head
        tokenizer: Tokenizer
        sample: Dict with 'intermediate_x', 'token_position', 'block_size_rel'
        device: torch device
    
    Returns:
        Dict with predictions and ground truth
    """
    # Extract data from sample
    # Note: intermediate_x has shape [1, seq_len] in JSON
    # Keep batch dimension for consistency with model input: [batch_size, seq_len]
    input_ids = torch.tensor(sample['intermediate_x'], dtype=torch.long).to(device)
    token_position = sample['features'][0][2]  # Position WITHIN generation (relative)
    ground_truth = sample['block_size_rel']
    
    # Calculate absolute position in full sequence (prompt + generation)
    gen_length = len(sample['full_token_ids'])
    seq_len = input_ids.shape[1]  # input_ids shape: [1, seq_len]
    prompt_len = seq_len - gen_length
    abs_pos = prompt_len + token_position  # Absolute position in full sequence
    
    # Run model forward pass
    model.eval()
    print(f"🔍 DEBUG inf_scheduler_head.py: input_ids.shape = {input_ids.shape}")
    print(f"🔍 DEBUG inf_scheduler_head.py: token_position = {token_position}, prompt_len = {prompt_len}, abs_pos = {abs_pos}")
    decoded_text = tokenizer.decode(input_ids[0].cpu().tolist(), skip_special_tokens=False)
    print(f"🔍 DEBUG inf_scheduler_head.py: decoded text = {decoded_text[:]}...")  # First 200 chars
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            token_positions=torch.tensor([abs_pos], device=device, dtype=torch.long),
            block_size_rel_labels=None  # No labels for inference
        )
    
    # Extract prediction at the absolute position
    predictions = outputs.block_size_predictions[0]  # [seq_len]
    predicted_value = predictions[abs_pos].item()
    
    return {
        'predicted': predicted_value,
        'ground_truth': ground_truth,
        'token_position': token_position,
        'abs_pos': abs_pos,
        'prompt_len': prompt_len,
        'seq_len': input_ids.shape[1],
        'error': abs(predicted_value - ground_truth),
        'error_pct': abs(predicted_value - ground_truth) / ground_truth * 100 if ground_truth != 0 else float('inf')
    }


def main():
    print("="*80)
    print("🔍 Scheduler Head Inference")
    print("="*80)
    
    # Configuration
    CONFIG = {
        'checkpoint_path': 'checkpoints/scheduler_head/scheduler_head_best.pt',
        'data_path': './data/sft_training_samples_greedy.json',
        'num_samples': 5,  # Test on first N samples
    }
    
    # Load model config
    print("\n[1/5] Loading model configuration...")
    model_config = torch.load('./cache/model_config.pt', weights_only=False)
    
    # Load base model with scheduler head
    print("\n[2/5] Loading base model with scheduler head...")
    model_args = type('Args', (), {
        'model_name_or_path': model_config.get('model_name_or_path'),
        'torch_dtype': 'bfloat16',
        'cache_dir': None,
        'model_revision': 'main',
        'token': None,
        'use_fast_tokenizer': True,
        'trust_remote_code': False
    })()
    
    model, tokenizer, device = load_model(model_args, use_custom=True)
    print(f"   Model loaded on: {device}")
    
    # Load trained scheduler head weights
    print("\n[3/5] Loading trained scheduler head weights...")
    model = load_trained_scheduler_head(model, CONFIG['checkpoint_path'])
    
    # Load test data
    print("\n[4/5] Loading test data...")
    with open(CONFIG['data_path'], 'r') as f:
        samples = json.load(f)
    print(f"✅ Loaded {len(samples)} samples")
    
    # Run inference
    print("\n[5/5] Running inference...")
    print("="*80)
    
    results = []
    for i, sample in enumerate(samples[:CONFIG['num_samples']]):
        result = run_inference_on_sample(model, tokenizer, sample, device)
        results.append(result)
        
        print(f"\n📊 Sample {i+1}:")
        print(f"   Predicted:      {result['predicted']:.6f}")
        print(f"   Ground Truth:   {result['ground_truth']:.6f}")
        print(f"   Absolute Error: {result['error']:.6f}")
        print(f"   Relative Error: {result['error_pct']:.2f}%")
        print(f"   Position (rel): {result['token_position']} (within generation)")
        print(f"   Position (abs): {result['abs_pos']} (in full sequence, prompt_len={result['prompt_len']})")
    
    # Summary statistics
    print("\n" + "="*80)
    print("📈 Summary Statistics:")
    print("="*80)
    
    avg_error = sum(r['error'] for r in results) / len(results)
    avg_error_pct = sum(r['error_pct'] for r in results) / len(results)
    max_error = max(r['error'] for r in results)
    min_error = min(r['error'] for r in results)
    
    print(f"   Average Absolute Error: {avg_error:.6f}")
    print(f"   Average Relative Error: {avg_error_pct:.2f}%")
    print(f"   Max Error: {max_error:.6f}")
    print(f"   Min Error: {min_error:.6f}")
    print(f"   Samples Tested: {len(results)}")
    
    # Check if overfitting was successful (very low error on training sample)
    if CONFIG['num_samples'] >= 1:
        first_sample_error = results[0]['error']
        print("\n" + "="*80)
        if first_sample_error < 0.01:
            print("✅ OVERFITTING SUCCESSFUL! Error < 0.01 on training sample")
        elif first_sample_error < 0.05:
            print("⚠️  Partial overfitting. Error < 0.05 but not < 0.01")
        else:
            print("❌ Overfitting failed. Error >= 0.05 on training sample")
        print("="*80)


if __name__ == '__main__':
    main()

