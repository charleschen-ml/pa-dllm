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
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import trl

# Import model loading function from run_inference
from run_inference import load_model


class SchedulerDataset(Dataset):
    """Dataset for training the scheduler head."""
    
    def __init__(self, json_path: str):
        """
        Load training samples from JSON.
        
        Each sample contains:
        - intermediate_x: [1, seq_len] token IDs (prompt + generated + masked)
        - features: [[step, confidence, position, ...], ...]
        - block_size_rel: float in [0, 1]
        
        We extract:
        - input_ids: intermediate_x[0] (squeeze to [seq_len])
        - token_position: features[0][2] (where to extract prediction)
        - label: block_size_rel
        """
        with open(json_path, 'r') as f:
            self.samples = json.load(f)
        
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
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
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
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, device):
    """Validate on validation set."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
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
    # Configuration
    CONFIG = {
        'data_path': 'data/sft_training_samples_greedy.json',
        'num_questions': 5,  # Number of questions to use (None = use all)
        'batch_size': 1,
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'val_split': 0,  # 10% for validation
        'checkpoint_dir': 'checkpoints/scheduler_head',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
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
    
    # 3. Load dataset
    print("\n[3/6] Loading dataset...")
    dataset = SchedulerDataset(CONFIG['data_path'])
    
    # Truncate dataset if num_questions is specified
    if CONFIG['num_questions'] is not None:
        original_size = len(dataset.samples)
        dataset.samples = dataset.samples[:CONFIG['num_questions']]
        print(f"📊 Truncated dataset from {original_size} to {len(dataset.samples)} samples")
    
    # Print first sample for inspection
    if len(dataset) > 0:
        first_sample = dataset[0]
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
    
    # Split into train/val
    val_size = int(CONFIG['val_split'] * len(dataset))
    train_size = len(dataset) - val_size
    
    # Handle edge case: if dataset is too small for split, use same data for both
    if val_size == 0:
        print("⚠️  Dataset too small for train/val split. Using same data for validation.")
        train_dataset = dataset
        val_dataset = dataset
    else:
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    print(f"✅ Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
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
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\n📍 Epoch {epoch}/{CONFIG['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"  Val Loss:   {val_loss:.4f}")
        
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
            best_path = os.path.join(CONFIG['checkpoint_dir'], 'scheduler_head_best.pt')
            torch.save({
                'epoch': epoch,
                'scheduler_head_state_dict': model.model.transformer.scheduler_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_path)
            print(f"  ⭐ Best model saved: {best_path}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Checkpoints saved in: {CONFIG['checkpoint_dir']}")
    print("="*60)


if __name__ == '__main__':
    main()

