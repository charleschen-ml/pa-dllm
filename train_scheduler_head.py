#!/usr/bin/env python3
"""
Train the scheduler head to predict block_size_rel.

This script:
1. Loads training data from data/sft_training_samples_greedy.json
2. Creates a PyTorch Dataset and DataLoader
3. Freezes the main LLaDA model weights
4. Trains only the scheduler_head using AdamW
5. Logs training metrics and saves checkpoints
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers without display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import trl

# Import model loading function from run_inference
from run_inference import load_model


class SchedulerDataset(Dataset):
    """Dataset for training the scheduler head."""
    
    def __init__(self, json_path: str, sample_indices: list = None):
        """
        Load training samples from JSON.
        
        Each sample contains:
        - intermediate_x: [1, seq_len] token IDs (prompt + generated + masked)
        - features: [[step, confidence, position, ...], ...]
        - block_size_rel: float in [0, 1]
        
        Args:
            json_path: Path to JSON file
            sample_indices: List of indices to use (for train/val split)
        
        We extract:
        - input_ids: intermediate_x[0] (squeeze to [seq_len])
        - token_position: features[0][2] (where to extract prediction)
        - label: block_size_rel
        """
        with open(json_path, 'r') as f:
            all_samples = json.load(f)
        
        if sample_indices is not None:
            self.samples = [all_samples[i] for i in sample_indices]
        else:
            self.samples = all_samples
        
        print(f"✅ Loaded {len(self.samples)} training samples from {json_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Extract input_ids (intermediate_x)
        # Note: intermediate_x has shape [1, seq_len] from generate_custom()
        # We squeeze out the batch dimension to get [seq_len]
        input_ids = sample['intermediate_x'][0]
        
        # Extract token_position (where to predict from)
        token_position = sample['features'][0][2]  # position in features
        
        # Extract label (block_size_rel)
        label = sample['block_size_rel']
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_position': token_position,
            'label': label
        }


def split_by_question(json_path: str, val_split: float, num_questions: int = None, seed: int = 42):
    """
    Split samples by question_id to avoid data leakage.
    
    Groups samples by their question_id, then splits questions into train/val.
    All samples from a question go to either train or val, never both.
    
    Args:
        json_path: Path to JSON file with samples
        val_split: Fraction of questions for validation (e.g., 0.2 = 20%)
        num_questions: Limit to first N questions (None = use all)
        seed: Random seed for reproducible splits
    
    Returns:
        train_indices, val_indices: Lists of sample indices for train and val
    """
    with open(json_path, 'r') as f:
        all_samples = json.load(f)
    
    # Group samples by question_id
    question_to_samples = {}
    for idx, sample in enumerate(all_samples):
        question_id = sample.get('question_id', idx)  # Fallback to index if no question_id
        
        if question_id not in question_to_samples:
            question_to_samples[question_id] = []
        question_to_samples[question_id].append(idx)
    
    # Get list of unique question IDs (sorted for consistency)
    question_ids = sorted(question_to_samples.keys())
    
    # Limit to first N questions if specified
    if num_questions is not None:
        question_ids = question_ids[:num_questions]
    
    print(f"\n📊 Question-Level Split:")
    print(f"   Total unique questions: {len(question_ids)}")
    print(f"   Total samples: {sum(len(question_to_samples[q]) for q in question_ids)}")
    
    # Shuffle questions with fixed seed
    rng = random.Random(seed)
    rng.shuffle(question_ids)
    
    # Split questions into train/val
    num_val_questions = max(1, int(len(question_ids) * val_split))
    val_question_ids = question_ids[:num_val_questions]
    train_question_ids = question_ids[num_val_questions:]
    
    # Get sample indices for train and val
    train_indices = []
    val_indices = []
    
    for q_id in train_question_ids:
        train_indices.extend(question_to_samples[q_id])
    
    for q_id in val_question_ids:
        val_indices.extend(question_to_samples[q_id])
    
    print(f"   Train: {len(train_question_ids)} questions ({len(train_indices)} samples)")
    print(f"   Val:   {len(val_question_ids)} questions ({len(val_indices)} samples)")
    print(f"   ✅ No data leakage - questions are fully separated")
    
    return train_indices, val_indices


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate batch samples with padding.
    
    Args:
        batch: List of dicts with 'input_ids', 'token_position', 'label'
    
    Returns:
        Dict with batched tensors:
        - input_ids: [batch_size, max_seq_len] (padded)
        - token_positions: [batch_size]
        - labels: [batch_size]
        - attention_mask: [batch_size, max_seq_len]
    """
    # Find max sequence length in batch
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    batch_size = len(batch)
    
    # Initialize tensors (pad with 0s)
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    token_positions = torch.zeros(batch_size, dtype=torch.long)
    labels = torch.zeros(batch_size, dtype=torch.float32)
    
    # Fill in actual values
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].size(0)
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = 1  # Mark non-padded positions
        token_positions[i] = item['token_position']
        labels[i] = item['label']
    
    return {
        'input_ids': input_ids,
        'token_positions': token_positions,
        'block_size_rel_labels': labels,
        'attention_mask': attention_mask
    }


def plot_training_curves(train_losses, val_losses, save_path):
    """
    Plot and save training/validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Add best epoch marker
    best_epoch = val_losses.index(min(val_losses)) + 1
    plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Training curve saved: {save_path}")
    plt.close()


def freeze_model_except_scheduler(model):
    """Freeze all parameters except scheduler_head."""
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze scheduler_head
    if hasattr(model.model.transformer, 'scheduler_head'):
        for param in model.model.transformer.scheduler_head.parameters():
            param.requires_grad = True
        print("✅ Unfroze scheduler_head parameters")
    else:
        raise ValueError("scheduler_head not found in model!")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    total_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        token_positions = batch['token_positions'].to(device)
        labels = batch['block_size_rel_labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Debug: Print batch info for first batch of first epoch
        if epoch == 1 and batch_idx == 0:
            print(f"\n🔍 Batch Debug Info:")
            print(f"   Batch size (actual): {input_ids.shape[0]}")
            print(f"   Sequence length (padded): {input_ids.shape[1]}")
            print(f"   Token positions: {token_positions.tolist()}")
            print(f"   Labels: {labels.tolist()}")
            
            # Show individual sequence lengths (before padding)
            print(f"\n   📏 Individual Sequence Lengths:")
            for i in range(input_ids.shape[0]):
                actual_len = attention_mask[i].sum().item()
                print(f"      Sample {i}: {actual_len} tokens (padded to {input_ids.shape[1]})")
            
            # Show attention mask for first 2 samples (to verify padding)
            print(f"\n   🎭 Attention Masks (first 2 samples):")
            for i in range(min(2, input_ids.shape[0])):
                mask = attention_mask[i].tolist()
                # Show where padding starts
                num_real = sum(mask)
                num_pad = len(mask) - num_real
                print(f"      Sample {i}: [{num_real} ones, {num_pad} zeros]")
                print(f"                  {mask[:20]}...{mask[-10:]}")
            
            # Verify token_positions are within valid range
            print(f"\n   ✅ Token Position Validation:")
            all_valid = True
            for i in range(input_ids.shape[0]):
                actual_len = attention_mask[i].sum().item()
                token_pos = token_positions[i].item()
                is_valid = token_pos < actual_len
                status = "✓" if is_valid else "✗ ERROR"
                print(f"      Sample {i}: pos={token_pos} < len={actual_len} [{status}]")
                if not is_valid:
                    all_valid = False
            if not all_valid:
                print(f"      ⚠️  WARNING: Some token positions are in padding region!")
            print()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_positions=token_positions,
            block_size_rel_labels=labels
        )
        
        loss = outputs.scheduler_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress every 20% of epoch (or at least every 10 batches)
        print_interval = max(10, total_batches // 5)
        if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == total_batches:
            progress = 100 * (batch_idx + 1) / total_batches
            print(f"    [{batch_idx + 1}/{total_batches}] {progress:.0f}% - Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, device):
    """Validate on validation set."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            token_positions = batch['token_positions'].to(device)
            labels = batch['block_size_rel_labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_positions=token_positions,
                block_size_rel_labels=labels
            )
            
            loss = outputs.scheduler_loss
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    # Set random seeds for reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # Make cudnn deterministic (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Configuration
    CONFIG = {
        # 'data_path': 'data/sft_training_samples_greedy.json',
        'data_path': 'data/sft_training_samples_multi_greedy_parallel.json',
        'num_questions': 100,  # Number of questions to use (None = use all)
        'batch_size': 1,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'val_split': 0.1,  # 10% for validation
        'checkpoint_dir': 'checkpoints/scheduler_head',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': SEED,
    }
    
    print("="*60)
    print("🚀 TRAINING SCHEDULER HEAD")
    print("="*60)
    print(f"Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Create checkpoint directory
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    
    # 1. Load model
    print("\n[1/6] Loading model...")
    model_config = torch.load('./cache/model_config.pt', weights_only=True)
    model_args = trl.ModelConfig(
        model_name_or_path=model_config['model_name_or_path'],
        trust_remote_code=True,
        torch_dtype='bfloat16'  # Use bfloat16 for memory efficiency (frozen model + small head)
    )
    
    # Load model using dispatcher (same as inference, sets .eval() mode)
    model, tokenizer, device = load_model(model_args, use_custom=True)
    # We'll switch to .train() mode later after freezing parameters
    
    # 2. Freeze model except scheduler_head
    print("\n[2/6] Freezing model weights (except scheduler_head)...")
    freeze_model_except_scheduler(model)
    
    # Switch to train mode (load_model_custom sets it to .eval())
    model.train()
    print("✅ Model set to training mode")
    
    # 3. Load dataset with question-level split (to avoid data leakage)
    print("\n[3/6] Loading dataset...")
    
    # Split by question to avoid data leakage
    train_indices, val_indices = split_by_question(
        CONFIG['data_path'],
        val_split=CONFIG['val_split'],
        num_questions=CONFIG['num_questions'],
        seed=CONFIG['seed']
    )
    
    # Create train and val datasets
    train_dataset = SchedulerDataset(CONFIG['data_path'], sample_indices=train_indices)
    val_dataset = SchedulerDataset(CONFIG['data_path'], sample_indices=val_indices)
    
    # Print first sample for inspection
    if len(train_dataset) > 0:
        first_sample = train_dataset[0]
        print(f"\n📋 First sample preview:")
        print(f"  input_ids shape: {first_sample['input_ids'].shape}")
        print(f"  token_position: {first_sample['token_position']}")
        print(f"  label (block_size_rel): {first_sample['label']:.4f}")
        
        # Decode the input_ids to show the actual text
        decoded_text = tokenizer.decode(first_sample['input_ids'], skip_special_tokens=False)
        print(f"\n  📝 Decoded input_ids (full, no truncation):")
        print(f"  {decoded_text}")
        
        # Get the attention mask from a collated batch
        dummy_batch = collate_fn([first_sample])
        print(f"\n  🎭 Attention mask (full, no truncation):")
        print(f"  {dummy_batch['attention_mask'][0].tolist()}")
    
    # 4. Create dataloaders
    print("\n[4/6] Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for debugging, increase for speed
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    print(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"   Gradient updates per epoch: {len(train_loader)}")
    print(f"   Total updates for {CONFIG['num_epochs']} epochs: {len(train_loader) * CONFIG['num_epochs']}")
    
    # 5. Setup optimizer
    print("\n[5/6] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate']
    )
    print(f"✅ Optimizer: AdamW with lr={CONFIG['learning_rate']}")
    
    # 6. Training loop
    print("\n[6/6] Starting training...")
    print("="*60)
    
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_epoch = 0
    
    # Track losses for plotting
    train_losses = []
    val_losses = []
    
    # Start total training timer
    total_start_time = time.time()
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\n📍 Epoch {epoch}/{CONFIG['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Track losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save checkpoint (only scheduler head, not full model)
        checkpoint_path = os.path.join(
            CONFIG['checkpoint_dir'], 
            f'scheduler_head_epoch_{epoch}.pt'
        )
        torch.save({
            'epoch': epoch,
            'scheduler_head_state_dict': model.model.transformer.scheduler_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"  💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model (only scheduler head)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            best_path = os.path.join(CONFIG['checkpoint_dir'], 'scheduler_head_best.pt')
            torch.save({
                'epoch': epoch,
                'scheduler_head_state_dict': model.model.transformer.scheduler_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_path)
            print(f"  ⭐ Best model saved: {best_path}")
    
    # Calculate total training time
    total_time = time.time() - total_start_time
    avg_epoch_time = total_time / CONFIG['num_epochs']
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print(f"📊 Best Results (Epoch {best_epoch}):")
    print(f"   Train Loss: {best_train_loss:.4f}")
    print(f"   Val Loss:   {best_val_loss:.4f}")
    print(f"\n⏱️  TIMING REPORT:")
    print(f"   Total training time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"   Average time per epoch: {avg_epoch_time:.1f}s")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Throughput: {len(train_loader.dataset) * CONFIG['num_epochs'] / total_time:.1f} samples/sec")
    print(f"💾 Checkpoints saved in: {CONFIG['checkpoint_dir']}")
    print("="*60)
    
    # Save loss history to JSON
    loss_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss,
        'timing': {
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'avg_epoch_time_seconds': avg_epoch_time,
            'throughput_samples_per_sec': len(train_loader.dataset) * CONFIG['num_epochs'] / total_time
        },
        'config': {k: v for k, v in CONFIG.items() if k not in ['device']}  # Exclude non-serializable
    }
    loss_history_path = os.path.join(CONFIG['checkpoint_dir'], 'loss_history.json')
    with open(loss_history_path, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"\n📝 Loss history saved: {loss_history_path}")
    
    # Plot training curves
    plot_path = os.path.join(CONFIG['checkpoint_dir'], 'training_curves.png')
    plot_training_curves(train_losses, val_losses, plot_path)


if __name__ == '__main__':
    main()

