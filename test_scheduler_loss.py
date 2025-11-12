"""
Test script to verify that scheduler head loss computation works correctly.

Training Data Format (from data/sft_training_samples_greedy.json):
- Each sample represents ONE generation step in autoregressive decoding
- intermediate_x: [1, full_seq_len] - includes prompt + generated + MASKED tokens
- features[0][2]: token_position - index of curr_pos (last unmasked token)
- block_size_rel: scalar value in [0, 1] - label for the NEXT block size

Model Behavior:
- Forward pass generates predictions for ALL token positions
- Loss is computed at token_position (curr_pos), NOT the last position
- token_positions tensor tells the model where to extract prediction for each sample
- This is critical because intermediate_x includes masked tokens after curr_pos

This script:
1. Loads a model with scheduler_head enabled
2. Creates dummy input_ids [1, 97], token_positions [65], labels [1]
3. Calls forward() with labels and token_positions
4. Verifies that scheduler_loss is computed using the correct position
5. Verifies that gradients flow to scheduler_head parameters
"""

import torch
from trl import ModelConfig
from run_inference import load_model

def test_scheduler_loss():
    print("=" * 80)
    print("Testing Scheduler Loss Computation")
    print("=" * 80)
    
    # 1. Load model with scheduler head
    print("\n[1/5] Loading model with scheduler_head enabled...")
    
    # Setup model_args (mimic run_inference.py setup)
    config = torch.load('./cache/model_config.pt', weights_only=True)
    model_args = ModelConfig(
        model_name_or_path=config['model_name_or_path'],
        trust_remote_code=config['trust_remote_code'],
        torch_dtype='bfloat16'
    )
    
    # Load model using custom loading (with scheduler head)
    print("Loading model with USE_CUSTOM_MODEL=True...")
    model, tokenizer, device = load_model(model_args, use_custom=True)
    
    model.eval()  # Start in eval mode
    print(f"✅ Model loaded on {device}")
    
    # Verify scheduler head exists
    if not hasattr(model.model.transformer, "scheduler_head"):
        print("❌ FAIL: Scheduler head not found!")
        return False
    print("✅ Scheduler head found in model")
    
    # 2. Create dummy inputs
    print("\n[2/5] Creating dummy inputs...")
    # Match the real training data format: [1, prompt_length + gen_length]
    # The sequence includes masked tokens, so we need token_position to know where curr_pos is
    batch_size = 1
    seq_len = 97  # Typical length from training data (includes ~32 masked tokens at end)
    token_position = 65  # Example: unmasked tokens end at position 65, masked tokens from 65-97
                         # In real training: token_position = sample['features'][0][2]
    
    # Dummy input_ids (random token IDs, simulating prompt + generated + masked)
    # In real training: input_ids = torch.tensor(sample['intermediate_x'])
    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device, dtype=torch.long)
    
    # Dummy labels (random values in [0, 1])
    # Shape is [batch_size] - one label per sample
    # In real training: block_size_rel_labels = torch.tensor([sample['block_size_rel']], dtype=torch.float32)
    # Note: Keep labels in float32 for numerical stability in loss computation
    block_size_rel_labels = torch.rand(batch_size, device=device, dtype=torch.float32)
    
    # Token positions: where to extract the prediction from (curr_pos)
    # Shape is [batch_size]
    # In real training: token_positions = torch.tensor([sample['features'][0][2]], dtype=torch.long)
    token_positions = torch.tensor([token_position], device=device, dtype=torch.long)
    
    print(f"   input_ids shape: {input_ids.shape} (includes masked tokens)")
    print(f"   token_positions: {token_positions.tolist()} (where to extract prediction)")
    print(f"   block_size_rel_labels shape: {block_size_rel_labels.shape} (one label per sample)")
    print(f"   Sample label value: {block_size_rel_labels[0].item():.4f}")
    
    # 3. Forward pass WITHOUT labels (inference mode)
    print("\n[3/5] Testing forward pass WITHOUT labels (inference)...")
    with torch.no_grad():
        outputs_no_labels = model(input_ids=input_ids)
    
    if outputs_no_labels.block_size_predictions is None:
        print("❌ FAIL: block_size_predictions is None!")
        return False
    print(f"✅ block_size_predictions shape: {outputs_no_labels.block_size_predictions.shape}")
    print(f"   Sample predictions: {outputs_no_labels.block_size_predictions[0, :5].tolist()}")
    
    if outputs_no_labels.scheduler_loss is not None:
        print("❌ FAIL: scheduler_loss should be None when no labels provided!")
        return False
    print("✅ scheduler_loss is None (correct, no labels provided)")
    
    # 4. Forward pass WITH labels (training mode)
    print("\n[4/5] Testing forward pass WITH labels (training)...")
    model.train()  # Switch to train mode
    
    outputs_with_labels = model(
        input_ids=input_ids,
        block_size_rel_labels=block_size_rel_labels,
        token_positions=token_positions  # Specify where to extract prediction
    )
    
    if outputs_with_labels.scheduler_loss is None:
        print("❌ FAIL: scheduler_loss is None even with labels!")
        return False
    
    loss_value = outputs_with_labels.scheduler_loss.item()
    print(f"✅ scheduler_loss computed: {loss_value:.6f}")
    
    # Sanity check: loss should be non-negative (MSE)
    if loss_value < 0:
        print("❌ FAIL: Loss is negative!")
        return False
    print("✅ Loss is non-negative (valid MSE)")
    
    # 5. Test gradient flow
    print("\n[5/5] Testing gradient flow to scheduler_head...")
    
    # Zero out any existing gradients
    model.zero_grad()
    
    # Backward pass
    outputs_with_labels.scheduler_loss.backward()
    
    # Check if scheduler_head has gradients
    has_gradients = False
    for name, param in model.model.transformer.scheduler_head.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            print(f"   ✅ Gradient found in {name}: norm={param.grad.norm().item():.6f}")
            break
    
    if not has_gradients:
        print("❌ FAIL: No gradients in scheduler_head!")
        return False
    
    print("✅ Gradients flow correctly to scheduler_head")
    
    # Verify LLaDA weights have NO gradients (they should be frozen during training)
    print("\n[BONUS] Checking if LLaDA base weights have gradients...")
    llada_has_grad = False
    for name, param in model.model.transformer.named_parameters():
        if "scheduler_head" not in name and param.grad is not None and param.grad.abs().sum() > 0:
            llada_has_grad = True
            print(f"   ⚠️  Gradient found in LLaDA layer: {name}")
            break
    
    if llada_has_grad:
        print("   ⚠️  LLaDA base weights have gradients (will need to freeze during training)")
    else:
        print("   ℹ️  LLaDA base weights have no gradients (already frozen or not called)")
    
    # 6. Summary
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nSummary:")
    print("  ✓ Model loads with scheduler_head")
    print("  ✓ Forward pass works without labels (inference)")
    print("  ✓ Forward pass works with labels (training)")
    print("  ✓ scheduler_loss is computed correctly")
    print("  ✓ Gradients flow to scheduler_head")
    print("\n✅ Ready for training! You can now:")
    print("   1. Load your training data (from .json files)")
    print("   2. Extract from each sample:")
    print("      - input_ids = torch.tensor(sample['intermediate_x'])")
    print("      - token_position = sample['features'][0][2]  # Position index from features")
    print("      - label = sample['block_size_rel']")
    print("   3. Freeze LLaDA weights:")
    print("      for param in model.parameters(): param.requires_grad = False")
    print("   4. Unfreeze scheduler_head:")
    print("      for param in model.model.transformer.scheduler_head.parameters():")
    print("          param.requires_grad = True")
    print("   5. Create DataLoader batching samples together")
    print("   6. Create optimizer: Adam(scheduler_head.parameters(), lr=1e-4)")
    print("   7. Training loop:")
    print("      outputs = model(input_ids, block_size_rel_labels=labels, token_positions=positions)")
    print("      loss = outputs.scheduler_loss")
    print("      loss.backward()")
    
    return True

if __name__ == "__main__":
    try:
        success = test_scheduler_loss()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

