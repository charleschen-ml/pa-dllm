#!/usr/bin/env python3
"""
Train the scheduler head to predict block_size_rel.

This script:
1. Loads training data from data/sft_training_samples_multi_greedy_parallel.csv
2. Creates a PyTorch Dataset and DataLoader
3. Freezes the main LLaDA model weights (optional - can train with features only)
4. Trains only the scheduler_head using AdamW
5. Logs training metrics and saves checkpoints
"""

import json
import os
import pandas as pd
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

# Import shared utilities
from scheduler_utils import (
    split_by_question, evaluate_classifier, print_metrics_summary,
    plot_confusion_matrix, plot_pr_curve
)


class SchedulerDataset(Dataset):
    """Dataset for training the scheduler head with XGBoost features."""
    
    # XGBoost features (30 total) - same as train_scheduler.py
    FEATURE_COLS = [
        'position_relative',
        # Confidence features (positions 0-9)
        'conf_0', 'conf_1', 'conf_2', 'conf_3', 'conf_4', 'conf_5', 'conf_6', 'conf_7', 'conf_8', 'conf_9',
        # Shannon entropy features (positions 0-9)
        'shannon_entropy_0', 'shannon_entropy_1', 'shannon_entropy_2', 'shannon_entropy_3', 'shannon_entropy_4',
        'shannon_entropy_5', 'shannon_entropy_6', 'shannon_entropy_7', 'shannon_entropy_8', 'shannon_entropy_9',
        # Aggregate features
        'top1_margin', 'mean_confidence', 'shannon_mean_entropy',
        'conf_std', 'shannon_entropy_std',
        'top4_conf_min', 'next4_conf_min', 'top8_conf_min', 'next8_conf_min'
    ]
    
    def __init__(self, csv_path: str, sample_indices: list = None, use_classification: bool = True):
        """
        Load training samples from CSV.
        
        Each row contains:
        - 30 XGBoost features (position, confidence, entropy stats)
        - block_size: target label (1 or 2 tokens for classification)
        
        Args:
            csv_path: Path to CSV file
            sample_indices: List of row indices to use (for train/val split)
            use_classification: If True, convert block_size to class labels (0=1 token, 1=2 tokens)
        
        We extract:
        - features: 30-dimensional feature vector (same as XGBoost)
        - labels: class labels if classification, else block_size_rel
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Select subset if sample_indices provided
        if sample_indices is not None:
            df = df.iloc[sample_indices].reset_index(drop=True)
        
        # Check for missing feature columns
        missing_cols = [col for col in self.FEATURE_COLS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in CSV: {missing_cols}")
        
        # Extract features and labels
        self.features = df[self.FEATURE_COLS].values  # shape: (num_samples, 30)
        
        if use_classification:
            # Convert block_size to class labels: 0 = 1 token, 1 = 2 tokens, 2 = 3 tokens, 3 = 4 tokens
            block_sizes = df['block_size'].values
            self.labels = block_sizes - 1  # block_size=1 → class 0, block_size=2 → class 1, etc.
            print(f"✅ Using 4-class classification (1, 2, 3, 4 tokens)")
            print(f"   Class distribution: {np.bincount(self.labels.astype(int))}")
        else:
            # Use block_size_rel for regression
            self.labels = df['block_size_rel'].values  # shape: (num_samples,)
            print(f"✅ Using regression (block_size_rel)")
        
        # Check for missing values and impute with median (same as XGBoost)
        missing_count = pd.isna(self.features).sum()
        if missing_count > 0:
            print(f"⚠️  Found {missing_count} missing values, imputing with column medians...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            self.features = imputer.fit_transform(self.features)
        
        print(f"✅ Loaded {len(self.features)} samples from {csv_path}")
        print(f"   Features shape: {self.features.shape} (30 XGBoost features)")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Return a single training sample.
        
        Returns:
            dict with:
            - features: (30,) numpy array of XGBoost features
            - label: float block_size_rel in [0, 1]
        """
        return {
            'features': self.features[idx],  # (30,) array
            'label': self.labels[idx]  # scalar
        }


# split_by_question is now imported from scheduler_utils


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate batch samples (no padding needed for fixed-size features).
    
    Args:
        batch: List of dicts with 'features' (30,) and 'label' (scalar)
    
    Returns:
        Dict with batched tensors:
        - features: [batch_size, 30] float tensor
        - labels: [batch_size] float tensor
    """
    batch_size = len(batch)
    
    # Stack features and labels
    features = torch.zeros(batch_size, 30, dtype=torch.float32)
    labels = torch.zeros(batch_size, dtype=torch.float32)
    
    for i, item in enumerate(batch):
        features[i] = torch.tensor(item['features'], dtype=torch.float32)
        labels[i] = item['label']
    
    return {
        'features': features,  # [batch_size, 30]
        'labels': labels  # [batch_size]
    }


class FeatureOnlySchedulerMLP(nn.Module):
    """
    Simple MLP that takes 30 XGBoost features and predicts block_size.
    
    This is a pure feature-based model (no transformer, no hidden states).
    Should match or beat XGBoost performance since it gets the same features.
    
    Supports both classification (4 classes: 1, 2, 3, 4 tokens) and regression (block_size_rel).
    """
    
    def __init__(self, input_dim=30, hidden_dims=[256, 128, 64], num_classes=None):
        """
        Args:
            input_dim: Number of input features (30 for XGBoost features)
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
    
    def forward(self, features):
        """
        Args:
            features: [batch_size, 30] tensor of XGBoost features
        
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


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, use_classification=False):
    """Train for one epoch with feature-only model."""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    total_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        features = batch['features'].to(device)  # [batch_size, 30]
        labels = batch['labels'].to(device)  # [batch_size]
        
        # Debug: Print batch info for first batch of first epoch
        if epoch == 1 and batch_idx == 0:
            print(f"\n🔍 Batch Debug Info:")
            print(f"   Batch size: {features.shape[0]}")
            print(f"   Feature dimensions: {features.shape[1]} (30 XGBoost features)")
            print(f"   Sample features[0]: {features[0, :5].tolist()}... (first 5)")
            print(f"   Sample labels: {labels[:5].tolist()}")
        
        # Forward pass
        outputs = model(features)  # [batch_size, num_classes] or [batch_size]
        
        if use_classification:
            # Classification: outputs are logits, labels are class indices
            labels = labels.long()  # Ensure labels are integers
            loss = criterion(outputs, labels)
        else:
            # Regression: outputs are predictions in [0, 1]
            loss = criterion(outputs, labels)
        
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


def validate(model, dataloader, criterion, device, use_classification=False):
    """Validate on validation set and return comprehensive metrics."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Collect all predictions and labels for metric computation
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            features = batch['features'].to(device)  # [batch_size, 30]
            labels = batch['labels'].to(device)  # [batch_size]
            
            # Forward pass
            outputs = model(features)  # [batch_size, num_classes] or [batch_size]
            
            if use_classification:
                # Classification: outputs are logits
                labels_long = labels.long()
                loss = criterion(outputs, labels_long)
                
                # Get predicted classes
                predicted_classes = torch.argmax(outputs, dim=1)  # [batch_size]
                all_preds.extend(predicted_classes.cpu().numpy().tolist())
                all_labels.extend(labels_long.cpu().numpy().tolist())
            else:
                # Regression: outputs are predictions in [0, 1]
                loss = criterion(outputs, labels)
                all_preds.extend(outputs.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
            
            total_loss += loss.item()
            num_batches += 1
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    if use_classification:
        # Classification metrics (same as XGBoost)
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Class distribution
        class_counts = np.bincount(all_preds.astype(int))
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'class_distribution': class_counts.tolist()
        }
    else:
        # Regression metrics
        mse = np.mean((all_preds - all_labels) ** 2)
        mae = np.mean(np.abs(all_preds - all_labels))
        
        # R² score
        ss_res = np.sum((all_labels - all_preds) ** 2)
        ss_tot = np.sum((all_labels - np.mean(all_labels)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        return {
            'loss': total_loss / num_batches,
            'mse': mse,
            'mae': mae,
            'r2': r2
        }


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
        # Data config
        'data_path': 'data/sft_training_samples_multi_greedy_parallel.csv',  # CSV with 30 XGBoost features
        'num_questions': None,  # Number of questions to use (None = use all)
        'val_split': 0.15,  # 10% for validation
        'test_split': 0.15,  # 20% for test
        
        # Model config
        'hidden_dims': [256, 128, 64],  # MLP hidden layer dimensions
        'use_classification': True,  # True = 4-class classification (1, 2, 3, 4 tokens), False = regression
        'num_classes': 4,  # 4-class classification: class 0 = 1 token, class 1 = 2 tokens, class 2 = 3 tokens, class 3 = 4 tokens
        
        # Training config
        'batch_size': 1,  # Larger batch size for feature-only model (no GPU memory for transformer)
        'num_epochs': 20,  # More epochs since it's fast without transformer
        'early_stopping_patience': 5,  # Be patient
        
        # Optimizer config
        'learning_rate': 1e-3,  # Higher LR for simple MLP
        'weight_decay': 0.1,  # L2 regularization for AdamW (0.0 = no regularization for overfit test)
        
        # System config
        'checkpoint_dir': 'checkpoints/scheduler_head_features',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': SEED,
    }
    
    print("="*60)
    print("🚀 TRAINING FEATURE-ONLY SCHEDULER MLP")
    print("   (Using 30 XGBoost features - No Transformer)")
    print("="*60)
    print(f"Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Create checkpoint directory
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    
    device = CONFIG['device']
    print(f"\n💻 Device: {device}")
    
    # 1. Create feature-only MLP model
    print("\n[1/5] Creating feature-only MLP...")
    use_classification = CONFIG.get('use_classification', False)
    num_classes = CONFIG.get('num_classes', None) if use_classification else None
    
    model = FeatureOnlySchedulerMLP(
        input_dim=30,  # 30 XGBoost features
        hidden_dims=CONFIG['hidden_dims'],
        num_classes=num_classes
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {total_params:,} parameters")
    if use_classification:
        print(f"   Architecture: 30 → {' → '.join(map(str, CONFIG['hidden_dims']))} → {num_classes} (classification)")
        print(f"   Task: 4-class classification (class 0 = 1 token, class 1 = 2 tokens, class 2 = 3 tokens, class 3 = 4 tokens)")
    else:
        print(f"   Architecture: 30 → {' → '.join(map(str, CONFIG['hidden_dims']))} → 1 (regression)")
        print(f"   Task: Regression (predict block_size_rel)")
    
    # 2. Load dataset with question-level split (to avoid data leakage)
    print("\n[2/5] Loading dataset...")
    
    # Split by question to avoid data leakage (train/val/test)
    train_indices, val_indices, test_indices = split_by_question(
        CONFIG['data_path'],
        val_split=CONFIG['val_split'],
        test_split=CONFIG['test_split'],
        num_questions=CONFIG['num_questions'],
        seed=CONFIG['seed']
    )
    
    # Create train, val, and test datasets
    train_dataset = SchedulerDataset(CONFIG['data_path'], sample_indices=train_indices, use_classification=use_classification)
    val_dataset = SchedulerDataset(CONFIG['data_path'], sample_indices=val_indices, use_classification=use_classification)
    test_dataset = SchedulerDataset(CONFIG['data_path'], sample_indices=test_indices, use_classification=use_classification)
    
    # Print first sample for inspection
    if len(train_dataset) > 0:
        first_sample = train_dataset[0]
        print(f"\n📋 First sample preview:")
        print(f"  Features shape: {len(first_sample['features'])}")
        print(f"  Features (first 5): {first_sample['features'][:5]}")
        print(f"  Label (block_size_rel): {first_sample['label']:.4f}")
    
    # 3. Create dataloaders
    print("\n[3/5] Creating dataloaders...")
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
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    print(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    print(f"   Gradient updates per epoch: {len(train_loader)}")
    print(f"   Total updates for {CONFIG['num_epochs']} epochs: {len(train_loader) * CONFIG['num_epochs']}")
    
    # 4. Setup optimizer and loss function
    print("\n[4/5] Setting up optimizer and loss...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    if use_classification:
        criterion = nn.CrossEntropyLoss()  # Cross-entropy for classification
        print(f"✅ Optimizer: AdamW (lr={CONFIG['learning_rate']}, weight_decay={CONFIG['weight_decay']})")
        print(f"✅ Loss function: CrossEntropyLoss (classification)")
    else:
        criterion = nn.MSELoss()  # Mean Squared Error for regression
        print(f"✅ Optimizer: AdamW (lr={CONFIG['learning_rate']}, weight_decay={CONFIG['weight_decay']})")
        print(f"✅ Loss function: MSE (regression)")
    
    # 5. Training loop
    print("\n[5/5] Starting training...")
    print("="*60)
    
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    # Track losses for plotting
    train_losses = []
    val_losses = []
    
    # Start total training timer
    total_start_time = time.time()
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\n📍 Epoch {epoch}/{CONFIG['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, use_classification=use_classification)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, use_classification=use_classification)
        val_loss = val_metrics['loss']
        print(f"  Val Loss:   {val_loss:.4f}")
        
        if use_classification:
            # Classification metrics
            print(f"  Val Accuracy:  {val_metrics['accuracy']:.4f}")
            print(f"  Val Precision: {val_metrics['precision']:.4f}")
            print(f"  Val Recall:    {val_metrics['recall']:.4f}")
            print(f"  Val F1:        {val_metrics['f1']:.4f}")
            print(f"  Val Pred Dist: {val_metrics['class_distribution']}")
        else:
            # Regression metrics
            print(f"  Val MSE:    {val_metrics['mse']:.4f} (RMSE: {np.sqrt(val_metrics['mse']):.4f})")
            print(f"  Val MAE:    {val_metrics['mae']:.4f}")
            print(f"  Val R²:     {val_metrics['r2']:.4f}")
            print(f"  [XGBoost baseline: RMSE=0.16, MAE=0.12, R²=0.63]")
        
        # Track losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            CONFIG['checkpoint_dir'], 
            f'mlp_epoch_{epoch}.pt'
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': CONFIG
        }, checkpoint_path)
        print(f"  💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_path = os.path.join(CONFIG['checkpoint_dir'], 'mlp_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': CONFIG
            }, best_path)
            print(f"  ⭐ Best model saved: {best_path}")
        else:
            epochs_without_improvement += 1
            print(f"  📊 No improvement for {epochs_without_improvement} epoch(s)")
            
            # Early stopping check
            if epochs_without_improvement >= CONFIG['early_stopping_patience']:
                print(f"\n🛑 Early stopping triggered! No improvement for {CONFIG['early_stopping_patience']} epochs.")
                print(f"   Best val loss: {best_val_loss:.4f} (Epoch {best_epoch})")
                break
    
    # Calculate total training time
    total_time = time.time() - total_start_time
    actual_epochs = epoch  # Use actual number of epochs completed
    avg_epoch_time = total_time / actual_epochs
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print(f"📊 Best Results (Epoch {best_epoch}):")
    print(f"   Train Loss: {best_train_loss:.4f}")
    print(f"   Val Loss:   {best_val_loss:.4f}")
    
    # ========================================
    # Final Evaluation on Train/Val/Test Sets
    # ========================================
    print("\n" + "="*80)
    print("📈 FINAL EVALUATION ON ALL SPLITS")
    print("="*80)
    
    # Load best model
    print(f"\n🔄 Loading best model (Epoch {best_epoch})...")
    best_path = os.path.join(CONFIG['checkpoint_dir'], 'mlp_best.pt')
    checkpoint = torch.load(best_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Best model loaded")
    
    # Helper function to collect predictions and labels
    def collect_predictions(dataloader):
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(features)  # [batch_size, num_classes] or [batch_size]
                
                if use_classification:
                    labels_long = labels.long()
                    predicted_classes = torch.argmax(outputs, dim=1)  # [batch_size]
                    probs = torch.softmax(outputs, dim=1)  # [batch_size, num_classes]
                    
                    all_labels.extend(labels_long.cpu().numpy().tolist())
                    all_preds.extend(predicted_classes.cpu().numpy().tolist())
                    all_probs.extend(probs.cpu().numpy())
                else:
                    all_labels.extend(labels.cpu().numpy().tolist())
                    all_preds.extend(outputs.cpu().numpy().tolist())
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        if use_classification:
            all_probs = np.array(all_probs)
        else:
            all_probs = None
        
        return all_labels, all_preds, all_probs
    
    # Evaluate on train, val, and test sets
    print("\n🔍 Evaluating on train set...")
    y_train_true, y_train_pred, y_train_probs = collect_predictions(train_loader)
    train_results = evaluate_classifier(
        y_train_true, y_train_pred, y_train_probs,
        num_classes=num_classes, split_name="Train"
    )
    
    print("🔍 Evaluating on val set...")
    y_val_true, y_val_pred, y_val_probs = collect_predictions(val_loader)
    val_results = evaluate_classifier(
        y_val_true, y_val_pred, y_val_probs,
        num_classes=num_classes, split_name="Val"
    )
    
    print("🔍 Evaluating on test set...")
    y_test_true, y_test_pred, y_test_probs = collect_predictions(test_loader)
    test_results = evaluate_classifier(
        y_test_true, y_test_pred, y_test_probs,
        num_classes=num_classes, split_name="Test"
    )
    
    # Print comprehensive metrics summary (same format as XGBoost)
    print_metrics_summary(
        train_results, val_results, test_results,
        decision_threshold=0.5,
        model_name="MLP"
    )
    
    # Plot confusion matrix and PR curve
    if use_classification:
        # Create output directory
        os.makedirs('./output', exist_ok=True)
        
        # Plot confusion matrix (test set)
        plot_confusion_matrix(
            y_test_true, y_test_pred, num_classes,
            save_path="./output/mlp_confusion_matrix.png"
        )
        
        # Plot PR curve for binary classification
        if num_classes == 2:
            plot_pr_curve(
                y_test_true, y_test_probs[:, 0],
                save_path="./output/mlp_pr_curve.png"
            )
    
    print("\n⏱️  TIMING REPORT:")
    print(f"   Epochs completed: {actual_epochs}/{CONFIG['num_epochs']}")
    print(f"   Total training time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"   Average time per epoch: {avg_epoch_time:.1f}s")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Throughput: {len(train_loader.dataset) * actual_epochs / total_time:.1f} samples/sec")
    print(f"💾 Checkpoints saved in: {CONFIG['checkpoint_dir']}")
    
    # Comparison with XGBoost baseline
    print("\n" + "="*80)
    print("📊 COMPARISON WITH XGBOOST BASELINE")
    print("="*80)
    print(f"MLP Test Accuracy:     {test_results['accuracy']:.3f} ({100*test_results['accuracy']:.1f}%)")
    print(f"XGBoost Test Accuracy: 0.668 (66.8%)  ← From output2.log")
    print()
    if test_results['accuracy'] > 0.668:
        improvement = (test_results['accuracy'] - 0.668) / 0.668 * 100
        print(f"✅ MLP is {improvement:.1f}% better than XGBoost!")
    elif test_results['accuracy'] < 0.668:
        degradation = (0.668 - test_results['accuracy']) / 0.668 * 100
        print(f"⚠️  MLP is {degradation:.1f}% worse than XGBoost")
    else:
        print(f"🟰 MLP matches XGBoost performance")
    print()
    print(f"Both models use the same 30 XGBoost features")
    print(f"Next step: Add LLM hidden states to MLP to improve performance")
    print("="*80)
    
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

