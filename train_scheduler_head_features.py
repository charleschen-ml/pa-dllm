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
    """Dataset for training the scheduler head with XGBoost features (and optionally hidden states)."""
    
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
    
    def __init__(self, data_path: str, sample_indices: list = None, use_classification: bool = True, 
                 data_format: str = 'csv', model=None, use_hidden_states: bool = False, csv_path: str = None):
        """
        Load training samples from CSV or JSON.
        
        Args:
            data_path: Path to CSV or JSON file
            sample_indices: List of indices to use (for train/val split)
            use_classification: If True, convert block_size to class labels
            data_format: 'csv' or 'json'
            model: LLaDA model for hidden state extraction (only if use_hidden_states=True)
            use_hidden_states: If True, extract hidden states from model
            csv_path: Optional CSV path for hybrid loading (when data_format='json')
        
        We extract:
        - features: 30-dimensional feature vector (XGBoost features)
        - hidden_states: 2048-dimensional if use_hidden_states=True
        - labels: class labels if classification, else block_size_rel
        """
        self.use_hidden_states = use_hidden_states
        self.model = model
        self.use_classification = use_classification
        self.csv_path = csv_path
        
        if data_format == 'csv':
            self._load_csv(data_path, sample_indices)
        elif data_format == 'json':
            self._load_json(data_path, sample_indices)
        else:
            raise ValueError(f"Unsupported data_format: {data_format}")
        
        # Validate
        if use_hidden_states and model is None:
            raise ValueError("model must be provided when use_hidden_states=True")
    
    def _load_csv(self, csv_path, sample_indices):
        """Load data from CSV format."""
        df = pd.read_csv(csv_path)
        
        if sample_indices is not None:
            df = df.iloc[sample_indices].reset_index(drop=True)
        
        # Check for missing feature columns
        missing_cols = [col for col in self.FEATURE_COLS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in CSV: {missing_cols}")
        
        # Extract features
        self.features = df[self.FEATURE_COLS].values  # (num_samples, 30)
        
        # Extract labels
        if self.use_classification:
            block_sizes = df['block_size'].values
            self.labels = block_sizes - 1  # class 0-3 for 1-4 tokens
            print(f"✅ Using 4-class classification (1, 2, 3, 4 tokens)")
            print(f"   Class distribution: {np.bincount(self.labels.astype(int))}")
        else:
            self.labels = df['block_size_rel'].values
            print(f"✅ Using regression (block_size_rel)")
        
        # Impute missing values
        missing_count = pd.isna(self.features).sum()
        if missing_count > 0:
            print(f"⚠️  Found {missing_count} missing values, imputing with column medians...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            self.features = imputer.fit_transform(self.features)
        
        # CSV doesn't have intermediate_x, so can't use hidden states
        if self.use_hidden_states:
            raise ValueError("CSV format doesn't support hidden states (no intermediate_x). Use JSON format.")
        
        self.input_ids = None
        self.token_positions = None
        
        print(f"✅ Loaded {len(self.features)} samples from {csv_path}")
        print(f"   Features shape: {self.features.shape} (30 XGBoost features)")
    
    def _load_json(self, json_path, sample_indices):
        """
        Load data from JSON format.
        
        Strategy: Load intermediate_x from JSON, but features from CSV (more complete).
        This is a hybrid approach since JSON has incomplete features.
        """
        import json
        
        # Determine CSV path from JSON path
        # Prefer explicit csv_path if provided, otherwise derive from JSON path
        if hasattr(self, 'csv_path') and self.csv_path:
            csv_path = self.csv_path
        else:
            csv_path = json_path.replace('.json', '.csv')
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        print(f"🔀 Hybrid loading: features from CSV, intermediate_x from JSON")
        print(f"   CSV: {csv_path}")
        print(f"   JSON: {json_path}")
        
        # Load both files
        with open(json_path, 'r') as f:
            json_samples = json.load(f)
        
        df_csv = pd.read_csv(csv_path)
        
        # Build mapping: (question_id, position) -> json_sample with intermediate_x
        json_map = {}
        for sample in json_samples:
            q_id = sample.get('question_id', -1)
            # Position from features[0][2]
            pos = sample['features'][0][2] if sample['features'] else 0
            json_map[(q_id, pos)] = sample
        
        print(f"   JSON samples: {len(json_samples)} (questions: {len(set(s.get('question_id', -1) for s in json_samples))})")
        print(f"   CSV samples: {len(df_csv)}")
        
        # Filter CSV to only samples that have matching JSON entries
        self.features = []
        self.labels = []
        self.input_ids = []
        self.token_positions = []
        matched_indices = []
        
        # Check which features are actually available in CSV
        available_features = [col for col in self.FEATURE_COLS if col in df_csv.columns]
        missing_features = [col for col in self.FEATURE_COLS if col not in df_csv.columns]
        
        if missing_features:
            print(f"⚠️  CSV missing {len(missing_features)}/{len(self.FEATURE_COLS)} features: {missing_features[:5]}...")
            print(f"   Using {len(available_features)} available features")
        
        for idx, row in df_csv.iterrows():
            # Skip if not in sample_indices
            if sample_indices is not None and idx not in sample_indices:
                continue
            
            q_id = row.get('question_id', -1)
            pos = row.get('position', 0)
            
            # Check if we have matching JSON entry
            if (q_id, pos) in json_map:
                # Extract available features from CSV, fill missing with 0.0
                xgb_features = []
                for col in self.FEATURE_COLS:
                    if col in df_csv.columns:
                        xgb_features.append(row[col])
                    else:
                        xgb_features.append(0.0)  # Fill missing features with 0
                self.features.append(xgb_features)
                
                # Extract label from CSV
                if self.use_classification:
                    block_size = row['block_size']
                    self.labels.append(block_size - 1)
                else:
                    self.labels.append(row['block_size_rel'])
                
                # Extract intermediate_x from JSON
                if self.use_hidden_states:
                    json_sample = json_map[(q_id, pos)]
                    # intermediate_x is [1, seq_len], squeeze to [seq_len]
                    self.input_ids.append(json_sample['intermediate_x'][0])
                    self.token_positions.append(int(pos))
                
                matched_indices.append(idx)
        
        if len(self.features) == 0:
            raise ValueError(f"No matching samples found between CSV and JSON! Check question_id alignment.")
        
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        
        # Check for missing values and impute
        missing_count = pd.isna(self.features).sum()
        if missing_count > 0:
            print(f"⚠️  Found {missing_count} missing values, imputing with column medians...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            self.features = imputer.fit_transform(self.features)
        
        if self.use_classification:
            print(f"✅ Using 4-class classification (1, 2, 3, 4 tokens)")
            print(f"   Class distribution: {np.bincount(self.labels.astype(int))}")
        else:
            print(f"✅ Using regression (block_size_rel)")
        
        print(f"✅ Loaded {len(self.features)} matched samples (CSV features + JSON intermediate_x)")
        print(f"   Features shape: {self.features.shape} (30 XGBoost features from CSV)")
        if self.use_hidden_states:
            print(f"   intermediate_x loaded from JSON for hidden state extraction")
    
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Return a single training sample.
        
        Returns:
            dict with:
            - features: (30,) or (2078,) numpy array
            - label: scalar
            - input_ids: (seq_len,) if use_hidden_states (for collate_fn)
            - token_position: int if use_hidden_states
        """
        result = {
            'features': self.features[idx],  # (30,) array
            'label': self.labels[idx]  # scalar
        }

        # Add data needed for hidden state extraction
        if self.use_hidden_states:
            result['input_ids'] = self.input_ids[idx]  # list of token IDs
            result['token_position'] = self.token_positions[idx]  # int
        
        return result


# split_by_question is imported from scheduler_utils (CSV only)
# We add a wrapper to support JSON

def split_by_question_flexible(data_path: str, data_format: str, val_split: float, 
                                test_split: float = None, num_questions: int = None, seed: int = 42):
    """
    Split by question for both CSV and JSON formats.
    
    Args:
        data_path: Path to CSV or JSON file
        data_format: 'csv' or 'json'
        val_split: Fraction for validation
        test_split: Fraction for test (optional)
        num_questions: Limit to N questions
        seed: Random seed
    
    Returns:
        (train_indices, val_indices, test_indices) if test_split provided
        (train_indices, val_indices) otherwise
    """
    if data_format == 'csv':
        # Use the scheduler_utils version for CSV
        return split_by_question(
            data_path, val_split=val_split, test_split=test_split,
            num_questions=num_questions, seed=seed
        )
    
    elif data_format == 'json':
        # Handle JSON format
        import json
        with open(data_path, 'r') as f:
            all_samples = json.load(f)
    
        # Group samples by question_id
        question_to_samples = {}
        for idx, sample in enumerate(all_samples):
            question_id = sample.get('question_id', idx)
            if question_id not in question_to_samples:
                question_to_samples[question_id] = []
            question_to_samples[question_id].append(idx)
        
        # Get sorted question IDs
        question_ids = sorted(question_to_samples.keys())
        
        # Limit to first N questions if specified
        if num_questions is not None:
            question_ids = question_ids[:num_questions]
        
        print(f"\n📊 Question-Level Split:")
        print(f"   Total unique questions: {len(question_ids)}")
        print(f"   Total samples: {sum(len(question_to_samples[q]) for q in question_ids)}")
        
        # Shuffle questions
        rng = random.Random(seed)
        rng.shuffle(question_ids)
        
        # Split questions
        if len(question_ids) == 1:
            print(f"   ⚠️  Only 1 question - using same question for all splits (overfit test)")
            train_question_ids = question_ids
            val_question_ids = question_ids
            test_question_ids = question_ids if test_split is not None else []
        else:
            if test_split is not None:
                # Three-way split
                num_test = max(1, int(len(question_ids) * test_split))
                num_val = max(1, int(len(question_ids) * val_split))
                
                test_question_ids = question_ids[:num_test]
                val_question_ids = question_ids[num_test:num_test + num_val]
                train_question_ids = question_ids[num_test + num_val:]
            else:
                # Two-way split
                num_val = max(1, int(len(question_ids) * val_split))
                val_question_ids = question_ids[:num_val]
                train_question_ids = question_ids[num_val:]
                test_question_ids = []
        
        # Get sample indices
        train_indices = []
        val_indices = []
        test_indices = []
        
        for q_id in train_question_ids:
            train_indices.extend(question_to_samples[q_id])
        for q_id in val_question_ids:
            val_indices.extend(question_to_samples[q_id])
        for q_id in test_question_ids:
            test_indices.extend(question_to_samples[q_id])
        
        print(f"   Train: {len(train_question_ids)} questions ({len(train_indices)} samples)")
        print(f"   Val:   {len(val_question_ids)} questions ({len(val_indices)} samples)")
        if test_split is not None:
            print(f"   Test:  {len(test_question_ids)} questions ({len(test_indices)} samples)")
        print(f"   ✅ No data leakage - questions are fully separated")
        
        if test_split is not None:
            return train_indices, val_indices, test_indices
        else:
            return train_indices, val_indices
    
    else:
        raise ValueError(f"Unsupported data_format: {data_format}")


def create_collate_fn(model=None, use_hidden_states=False, device='cuda', use_dual_stream=False):
    """
    Factory function to create a collate function with optional hidden state extraction.
    
    Args:
        model: LLaDA model for hidden state extraction
        use_hidden_states: Whether to extract hidden states
        device: Device for model inference
        use_dual_stream: If True, return full hidden states separately (not concatenated)
    
    Returns:
        collate_fn function
    """
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch samples with optional hidden state extraction.
        
        Args:
            batch: List of dicts with 'features', 'label', and optionally 'input_ids', 'token_position'
        
        Returns:
            Dict with batched tensors:
                - features: [batch_size, 30] or [batch_size, 2078] float tensor
                - labels: [batch_size] float tensor
        """
        batch_size = len(batch)
        
        # Extract XGBoost features (30-dim)
        xgb_features = torch.zeros(batch_size, 30, dtype=torch.float32)
        labels = torch.zeros(batch_size, dtype=torch.float32)
        
        # Debug flag to print once
        if not hasattr(collate_fn, '_debug_printed'):
            collate_fn._debug_printed = False
        
        for i, item in enumerate(batch):
            xgb_features[i] = torch.tensor(item['features'], dtype=torch.float32)
            labels[i] = float(item['label'])  # Convert numpy.float32 to Python float
        
        # If using hidden states, extract them dynamically
        if use_hidden_states and model is not None:
                # Prepare input_ids with padding
                input_ids_list = [item['input_ids'] for item in batch]
                token_positions = [item['token_position'] for item in batch]
                
                # Pad sequences to max length in batch
                max_len = max(len(ids) for ids in input_ids_list)
                input_ids_padded = []
                attention_masks = []
                
                for ids in input_ids_list:
                    # Pad to max_len
                    padding_length = max_len - len(ids)
                    padded_ids = ids + [0] * padding_length  # Assume 0 is pad token
                    attention_mask = [1] * len(ids) + [0] * padding_length
                    
                    input_ids_padded.append(padded_ids)
                    attention_masks.append(attention_mask)
                
                # Convert to tensors
                input_ids_tensor = torch.tensor(input_ids_padded, dtype=torch.long).to(device)
                attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long).to(device)
                
                # Extract hidden states from model
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids_tensor,
                        attention_mask=attention_mask_tensor,
                        output_hidden_states=True
                    )
                    # Get last layer hidden states: [batch_size, seq_len, hidden_dim]
                    last_hidden_states = outputs.hidden_states[-1]
                    
                    # For dual-stream mode: return FULL hidden states [batch_size, seq_len, hidden_dim]
                    # For single-stream mode: extract single position [batch_size, hidden_dim]
                    
                    if use_dual_stream:
                        # Return full hidden states (will be pooled by the model)
                        # Convert BFloat16 to Float32 for MLP compatibility
                        hidden_states_tensor = last_hidden_states.float().cpu()  # [batch_size, seq_len, hidden_dim]
                        
                        # Debug: Print extraction info once
                        if not collate_fn._debug_printed:
                            print(f"\n🔍 Hidden State Extraction Debug (first batch) - DUAL STREAM MODE:")
                            print(f"   Full hidden states shape: {hidden_states_tensor.shape}")
                            print(f"   Will be pooled by the model after projection")
                            print(f"   ✅ Using ALL positions for dual-stream!")
                            collate_fn._debug_printed = True
                    else:
                        # Extract hidden state at generation position for each sample
                        # IMPORTANT: Position in features is RELATIVE to generation start,
                        # but we need to find the ABSOLUTE position in the sequence.
                        # Strategy: Find first <|mdm_mask|> token (ID 126336) - that's where generation is at this step
                        MDM_MASK_TOKEN_ID = 126336
                        hidden_states_batch = []
                        
                        for i, pos in enumerate(token_positions):
                            # Find where generation is happening (first mask token)
                            sample_input_ids = input_ids_padded[i]  # Full token ID list
                            
                            # Find first occurrence of mask token
                            gen_position = None
                            for j, token_id in enumerate(sample_input_ids):
                                if token_id == MDM_MASK_TOKEN_ID:
                                    gen_position = j
                                    break
                            
                            # If no mask found, fall back to using attention mask length
                            if gen_position is None:
                                # Use last valid token position
                                valid_length = sum(attention_masks[i])
                                gen_position = max(0, valid_length - 1)
                            
                            # Extract hidden state at generation position
                            gen_position = min(gen_position, last_hidden_states.shape[1] - 1)
                            hidden_state = last_hidden_states[i, gen_position, :]  # [hidden_dim]
                            hidden_states_batch.append(hidden_state)
                        
                        # Stack: [batch_size, hidden_dim]
                        # Convert BFloat16 to Float32 for MLP compatibility
                        hidden_states_tensor = torch.stack(hidden_states_batch).float().cpu()
                        
                        # Debug: Print extraction info once
                        if not collate_fn._debug_printed:
                            print(f"\n🔍 Hidden State Extraction Debug (first batch) - SINGLE STREAM MODE:")
                            print(f"   Sample 0:")
                            print(f"     Relative position from CSV: {token_positions[0]}")
                            print(f"     Sequence length: {len(input_ids_padded[0])}")
                            print(f"     First mask index (gen_position): {gen_position}")
                            print(f"     Hidden state extracted from index: {gen_position}")
                            print(f"   ✅ Now extracting from generation position, not prompt!")
                            collate_fn._debug_printed = True
                
                if use_dual_stream:
                    # Return separate features and hidden states
                    result = {
                        'features': xgb_features,  # [batch_size, 30]
                        'hidden_states': hidden_states_tensor,  # [batch_size, seq_len, hidden_dim]
                        'labels': labels  # [batch_size]
                    }
                else:
                    # Concatenate XGBoost features + hidden states: [batch_size, 30 + 2048]
                    features = torch.cat([xgb_features, hidden_states_tensor], dim=1)
                    result = {
                        'features': features,  # [batch_size, 30 + hidden_dim]
                        'labels': labels  # [batch_size]
                    }
        else:
            # Use only XGBoost features
            result = {
                'features': xgb_features,  # [batch_size, 30]
                'labels': labels  # [batch_size]
            }
        
        return result
    
    return collate_fn


class DualStreamSchedulerMLP(nn.Module):
    """
    Dual-stream MLP with per-class learnable weights.
    
    Architecture:
    - Stream 1: Hidden states (97, 4096) → project → pool → MLP → [4] logits
    - Stream 2: XGBoost features (30) → MLP → [4] logits
    - Fusion: Per-class weighted sum with learnable weights
    
    This allows the model to learn how much to trust each stream for each class.
    """
    
    def __init__(self, xgb_feature_dim=30, hidden_size=4096, projection_dim=512, 
                 mlp_hidden_dims=[256, 128], num_classes=4):
        """
        Args:
            xgb_feature_dim: Number of XGBoost features (default: 30)
            hidden_size: Hidden state dimension from transformer (default: 4096)
            projection_dim: Dimension to project hidden states to before pooling (default: 512)
            mlp_hidden_dims: Hidden dimensions for both MLPs (default: [256, 128])
            num_classes: Number of output classes (default: 4)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        
        # ===== Hidden State Stream =====
        # Process ALL positions through MLP, pool at the very end (like LM head)
        # nn.Linear operates on last dimension, so this processes each position independently
        hidden_layers = []
        prev_dim = hidden_size  # Start from 4096
        
        # First project down
        hidden_layers.extend([
            nn.Linear(prev_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        ])
        prev_dim = projection_dim
        
        # Then process through MLP layers
        for hidden_dim in mlp_hidden_dims:
            hidden_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final layer: output num_classes for each position
        hidden_layers.append(nn.Linear(prev_dim, num_classes))
        self.hidden_mlp = nn.Sequential(*hidden_layers)
        # After this MLP: [batch, 97, 4096] → [batch, 97, 4]
        # We'll pool to [batch, 4] in forward()
        
        # ===== Feature Stream =====
        feature_layers = []
        prev_dim = xgb_feature_dim
        for hidden_dim in mlp_hidden_dims:
            feature_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        feature_layers.append(nn.Linear(prev_dim, num_classes))
        self.feature_mlp = nn.Sequential(*feature_layers)
        
        # ===== Per-Class Fusion Weights =====
        # Initialize to 0.5 (equal weighting)
        self.class_weights = nn.Parameter(torch.ones(num_classes) * 0.0)  # sigmoid(0) = 0.5
    
    def forward(self, xgb_features, hidden_states=None):
        """
        Args:
            xgb_features: [batch_size, 30] XGBoost features
            hidden_states: [batch_size, seq_len, 4096] transformer hidden states (optional)
        
        Returns:
            [batch_size, num_classes] logits
        """
        if hidden_states is None:
            # Features-only mode (fallback)
            return self.feature_mlp(xgb_features)
        
        # ===== Process Hidden States =====
        # Convert BFloat16 from LLaDA model to Float32 for MLP
        if hidden_states.dtype == torch.bfloat16:
            hidden_states = hidden_states.float()
        
        # Process ALL positions through MLP (just like LM head does)
        # nn.Linear operates on last dimension, so each position processed independently
        # [B, seq_len, 4096] → [B, seq_len, 512] → ... → [B, seq_len, 4]
        h = self.hidden_mlp(hidden_states)
        
        # Pool at the VERY END (after all transformations)
        # [B, seq_len, 4] → [B, 4]
        hidden_logits = h.mean(dim=1)
        
        # ===== Process XGBoost Features =====
        # [B, 30] → [B, num_classes]
        feature_logits = self.feature_mlp(xgb_features)
        
        # ===== Per-Class Weighted Fusion =====
        # Normalize weights to [0, 1]
        w_h = torch.sigmoid(self.class_weights)  # [num_classes]
        w_f = 1 - w_h  # [num_classes]
        
        # Element-wise weighted sum
        # w_h broadcasts from [4] to [B, 4]
        final_logits = w_h * hidden_logits + w_f * feature_logits
        
        return final_logits
    
    def get_fusion_weights(self):
        """Return interpretable fusion weights for each class."""
        w_h = torch.sigmoid(self.class_weights).detach().cpu().numpy()
        w_f = 1 - w_h
        
        weights_dict = {}
        for i in range(self.num_classes):
            weights_dict[f'class_{i+1}_tokens'] = {
                'hidden_weight': float(w_h[i]),
                'feature_weight': float(w_f[i]),
                'preference': 'hidden' if w_h[i] > 0.6 else ('feature' if w_f[i] > 0.6 else 'balanced')
            }
        return weights_dict


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


def plot_training_curves(train_losses, val_losses, val_accuracies=None, save_path='training_curves.png'):
    """
    Plot and save training/validation loss and accuracy curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_accuracies: List of validation accuracies per epoch (optional)
        save_path: Path to save the plot
    """
    if val_accuracies is not None:
        # Create 2-subplot figure: loss + accuracy
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        epochs = range(1, len(train_losses) + 1)
        
        # Plot 1: Loss
        ax1.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
        ax1.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Add best loss epoch marker
        best_loss_epoch = val_losses.index(min(val_losses)) + 1
        ax1.axvline(x=best_loss_epoch, color='g', linestyle='--', alpha=0.5, 
                    label=f'Best Val Loss (Epoch {best_loss_epoch})')
        ax1.legend(fontsize=10)
        
        # Plot 2: Validation Accuracy
        ax2.plot(epochs, val_accuracies, 'r-s', label='Val Accuracy', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Add best accuracy epoch marker
        best_acc_epoch = val_accuracies.index(max(val_accuracies)) + 1
        ax2.axvline(x=best_acc_epoch, color='purple', linestyle='--', alpha=0.5,
                    label=f'Best Val Acc (Epoch {best_acc_epoch})')
        ax2.legend(fontsize=10)
        
        # Highlight discrepancy if epochs differ
        if best_loss_epoch != best_acc_epoch:
            ax2.axvline(x=best_loss_epoch, color='g', linestyle=':', alpha=0.3,
                        label=f'Best Loss Epoch ({best_loss_epoch})')
            ax2.legend(fontsize=9)
        
        plt.tight_layout()
    else:
        # Original single plot (loss only)
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
        plt.plot(epochs, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
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


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, use_classification=False, use_dual_stream=False):
    """Train for one epoch (supports both single-stream and dual-stream models)."""
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
            if 'hidden_states' in batch:
                print(f"   Hidden state dimensions: {batch['hidden_states'].shape} (seq_len x hidden_size)")
            print(f"   Sample features[0]: {features[0, :5].tolist()}... (first 5)")
            print(f"   Sample labels: {labels[:5].tolist()}")
        
        # Forward pass
        if use_dual_stream:
            # Dual-stream model: pass both features and hidden states
            hidden_states = batch['hidden_states'].to(device) if 'hidden_states' in batch else None
            outputs = model(features, hidden_states)  # [batch_size, num_classes]
        else:
            # Single-stream model: only features
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


def validate(model, dataloader, criterion, device, use_classification=False, use_dual_stream=False):
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
            if use_dual_stream:
                # Dual-stream model: pass both features and hidden states
                hidden_states = batch['hidden_states'].to(device) if 'hidden_states' in batch else None
                outputs = model(features, hidden_states)  # [batch_size, num_classes]
            else:
                # Single-stream model: only features
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
        'data_format': 'json',  # 'csv' or 'json' (JSON needed for hidden states)
        'data_path': 'data/sft_training_samples_multi_greedy_parallel.json',  # Path to data
        'csv_path': 'data/sft_training_samples_multi_greedy_parallel.csv',  # CSV for features (hybrid mode: ALL 30 FEATURES NOW INCLUDED!)
        'num_questions': 10,  # Number of questions to use (None = use all, 1 = overfit test)
        'val_split': 0.15,  # 15% for validation
        'test_split': 0.15,  # 15% for test
        
        # Model config
        'use_dual_stream': True,  # ⭐ NEW: Use dual-stream architecture (hidden states + features with per-class weights)
        'hidden_dims': [256, 128],  # MLP hidden layer dimensions (for both streams)
        'projection_dim': 512,  # Project hidden states from 4096 → 512 before pooling
        'use_classification': True,  # True = 4-class classification (1, 2, 3, 4 tokens), False = regression
        'num_classes': 4,  # 4-class classification: class 0 = 1 token, class 1 = 2 tokens, class 2 = 3 tokens, class 3 = 4 tokens
        'use_hidden_states': True,  # ⭐ REQUIRED for dual-stream: Extract LLaDA hidden states
        
        # Training config
        'batch_size': 4,  # Smaller batch when using hidden states (more memory)
        'num_epochs': 1,  # More epochs to learn fusion weights
        'early_stopping_patience': 10,  # Be patient with dual-stream
        'use_class_weights': True,  # Balance classes (CRITICAL for imbalanced data!)
        
        # Optimizer config
        'learning_rate': 1e-3,  # Higher LR for simple MLP
        'weight_decay': 0.01,  # L2 regularization for AdamW
        
        # System config
        'checkpoint_dir': 'checkpoints/scheduler_head_dual_stream',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': SEED,
    }
    
    use_hidden_states = CONFIG.get('use_hidden_states', False)
    data_format = CONFIG.get('data_format', 'csv')
    
    print("="*60)
    if use_hidden_states:
        print("🚀 TRAINING HYBRID SCHEDULER MLP")
        print("   (30 XGBoost features + LLaDA hidden states)")
    else:
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
    
    # 0. Optionally load LLaDA model for hidden state extraction
    llada_model = None
    hidden_size = 0
    if use_hidden_states:
        print("\n[0/5] Loading LLaDA model for hidden state extraction...")
        model_config = torch.load('./cache/model_config.pt', weights_only=True)
        model_args = trl.ModelConfig(
            model_name_or_path=model_config['model_name_or_path'],
            trust_remote_code=True,
            torch_dtype='bfloat16'  # Use bfloat16 for memory efficiency
        )
        
        # Load model (frozen, eval mode)
        llada_model, tokenizer, _ = load_model(model_args, use_custom=False)
        llada_model.eval()  # Ensure eval mode
        
        # Freeze all parameters
        for param in llada_model.parameters():
            param.requires_grad = False
        
        # Get actual hidden size from model
        hidden_size = llada_model.config.hidden_size
        
        print(f"✅ LLaDA model loaded (frozen, eval mode)")
        print(f"   Model: {model_config['model_name_or_path']}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Model will extract hidden states during training")
    
    # 1. Create MLP model
    print("\n[1/5] Creating MLP...")
    use_classification = CONFIG.get('use_classification', False)
    num_classes = CONFIG.get('num_classes', None) if use_classification else None
    use_dual_stream = CONFIG.get('use_dual_stream', False)
    
    if use_dual_stream:
        # Dual-stream architecture: hidden states + features with per-class fusion
        projection_dim = CONFIG.get('projection_dim', 512)
        
        mlp_model = DualStreamSchedulerMLP(
            xgb_feature_dim=30,
            hidden_size=hidden_size,
            projection_dim=projection_dim,
            mlp_hidden_dims=CONFIG['hidden_dims'],
            num_classes=num_classes
    ).to(device)
        
        print(f"✅ Dual-Stream MLP created")
        print(f"   Stream 1 (Hidden): [seq_len, {hidden_size}] → MLP[{hidden_size}→{projection_dim}→{CONFIG['hidden_dims']}→4] → pool → [4]")
        print(f"   Stream 2 (Features): [30] → MLP{CONFIG['hidden_dims']} → [4]")
        print(f"   Fusion: Per-class weighted sum (4 learnable weights)")
        print(f"   ⭐ Hidden stream pools LATE (after all MLP layers, like LM head)")
        
    else:
        # Single-stream architecture (original)
        # Determine input dimension
        if use_hidden_states:
            input_dim = 30 + hidden_size  # XGBoost features + hidden states
            print(f"   Input dimension: 30 (features) + {hidden_size} (hidden states) = {input_dim}")
        else:
            input_dim = 30  # Only XGBoost features
        
        mlp_model = FeatureOnlySchedulerMLP(
            input_dim=input_dim,
            hidden_dims=CONFIG['hidden_dims'],
            num_classes=num_classes
        ).to(device)
        
        print(f"✅ MLP created")
        if use_classification:
            print(f"   Architecture: {input_dim} → {' → '.join(map(str, CONFIG['hidden_dims']))} → {num_classes} (classification)")
            print(f"   Task: 4-class classification (class 0 = 1 token, class 1 = 2 tokens, class 2 = 3 tokens, class 3 = 4 tokens)")
        else:
            print(f"   Architecture: {input_dim} → {' → '.join(map(str, CONFIG['hidden_dims']))} → 1 (regression)")
            print(f"   Task: Regression (predict block_size_rel)")
    
    # Count parameters
    total_params = sum(p.numel() for p in mlp_model.parameters())
    trainable_params = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # 2. Load dataset with question-level split (to avoid data leakage)
    print("\n[2/5] Loading dataset...")
    
    # Split by question to avoid data leakage (train/val/test)
    train_indices, val_indices, test_indices = split_by_question_flexible(
        CONFIG['data_path'],
        data_format=data_format,
        val_split=CONFIG['val_split'],
        test_split=CONFIG['test_split'],
        num_questions=CONFIG['num_questions'],
        seed=CONFIG['seed']
    )
    
    # Create train, val, and test datasets
    train_dataset = SchedulerDataset(
        CONFIG['data_path'], 
        sample_indices=train_indices, 
        use_classification=use_classification,
        data_format=data_format,
        model=llada_model,
        use_hidden_states=use_hidden_states,
        csv_path=CONFIG.get('csv_path', None)
    )
    val_dataset = SchedulerDataset(
        CONFIG['data_path'], 
        sample_indices=val_indices, 
        use_classification=use_classification,
        data_format=data_format,
        model=llada_model,
        use_hidden_states=use_hidden_states,
        csv_path=CONFIG.get('csv_path', None)
    )
    test_dataset = SchedulerDataset(
        CONFIG['data_path'], 
        sample_indices=test_indices, 
        use_classification=use_classification,
        data_format=data_format,
        model=llada_model,
        use_hidden_states=use_hidden_states,
        csv_path=CONFIG.get('csv_path', None)
    )
    
    # Print first sample for inspection
    if len(train_dataset) > 0:
        first_sample = train_dataset[0]
        print(f"\n📋 First sample preview:")
        print(f"  Features shape: {len(first_sample['features'])}")
        print(f"  Features (first 5): {first_sample['features'][:5]}")
        print(f"  Label (block_size_rel): {first_sample['label']:.4f}")
    
    # 3. Create dataloaders
    print("\n[3/5] Creating dataloaders...")
    
    # Create collate function (with or without hidden state extraction)
    collate_fn = create_collate_fn(
        model=llada_model,
        use_hidden_states=use_hidden_states,
        device=device,
        use_dual_stream=use_dual_stream
    )
    
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
        mlp_model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    if use_classification:
        # Optionally compute class weights to handle imbalance (same as XGBoost)
        if CONFIG.get('use_class_weights', True):
            from sklearn.utils.class_weight import compute_class_weight
            
            # Get class distribution from training set
            train_labels = np.array([train_dataset[i]['label'] for i in range(len(train_dataset))])
            unique_classes = np.arange(num_classes)
            
            # Compute balanced class weights
            class_weights_np = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
            class_weights = torch.FloatTensor(class_weights_np).to(device)
            
            # Show class weights
            print(f"\n⚖️  Computing class weights for imbalanced data...")
            for class_idx in range(num_classes):
                block_size = class_idx + 1  # class_to_blocksize
                token_str = f"{block_size} token" + ("s" if block_size > 1 else "")
                print(f"  Class {class_idx} ({token_str:>8}): weight = {class_weights_np[class_idx]:.2f}")
            
            criterion = nn.CrossEntropyLoss(weight=class_weights)  # Cross-entropy with class weights
            print(f"\n✅ Optimizer: AdamW (lr={CONFIG['learning_rate']}, weight_decay={CONFIG['weight_decay']})")
            print(f"✅ Loss function: CrossEntropyLoss (classification with class weights)")
        else:
            print(f"\n⚠️  Class weights disabled - training with uniform weights")
            criterion = nn.CrossEntropyLoss()  # Cross-entropy without class weights
            print(f"✅ Optimizer: AdamW (lr={CONFIG['learning_rate']}, weight_decay={CONFIG['weight_decay']})")
            print(f"✅ Loss function: CrossEntropyLoss (classification, NO class weights)")
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
    
    # Track losses and accuracies for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []  # Track val accuracy to visualize loss vs accuracy discrepancy
    
    # Start total training timer
    total_start_time = time.time()
    
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"\n📍 Epoch {epoch}/{CONFIG['num_epochs']}")
        
        # Train
        train_loss = train_epoch(mlp_model, train_loader, optimizer, criterion, device, epoch, 
                                use_classification=use_classification, use_dual_stream=use_dual_stream)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = validate(mlp_model, val_loader, criterion, device, 
                              use_classification=use_classification, use_dual_stream=use_dual_stream)
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
        
        # Track losses and accuracies
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if use_classification:
            val_accuracies.append(val_metrics['accuracy'])
        
        # Save checkpoint
        checkpoint_path = os.path.join(
            CONFIG['checkpoint_dir'], 
            f'mlp_epoch_{epoch}.pt'
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': mlp_model.state_dict(),
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
                'model_state_dict': mlp_model.state_dict(),
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
    mlp_model.load_state_dict(checkpoint['model_state_dict'])
    mlp_model.eval()
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
                
                # Forward pass (dual-stream or single-stream)
                if use_dual_stream:
                    hidden_states = batch['hidden_states'].to(device) if 'hidden_states' in batch else None
                    outputs = mlp_model(features, hidden_states)
                else:
                    outputs = mlp_model(features)  # [batch_size, num_classes] or [batch_size]
                
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
    
    # Print learned fusion weights for dual-stream model
    if use_dual_stream:
        print("\n" + "="*80)
        print("🎯 LEARNED FUSION WEIGHTS (Per-Class)")
        print("="*80)
        fusion_weights = mlp_model.get_fusion_weights()
        for class_name, weights in fusion_weights.items():
            print(f"\n{class_name}:")
            print(f"   Hidden states: {weights['hidden_weight']:.1%}")
            print(f"   XGBoost features: {weights['feature_weight']:.1%}")
            print(f"   → Preference: {weights['preference'].upper()}")
        
        # Summary
        avg_hidden_weight = np.mean([w['hidden_weight'] for w in fusion_weights.values()])
        avg_feature_weight = np.mean([w['feature_weight'] for w in fusion_weights.values()])
        print(f"\n📊 Average across all classes:")
        print(f"   Hidden states: {avg_hidden_weight:.1%}")
        print(f"   XGBoost features: {avg_feature_weight:.1%}")
    
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
    
    # Plot training curves (loss and accuracy)
    plot_path = os.path.join(CONFIG['checkpoint_dir'], 'training_curves.png')
    val_accs_to_plot = val_accuracies if (use_classification and len(val_accuracies) > 0) else None
    plot_training_curves(train_losses, val_losses, val_accs_to_plot, plot_path)


if __name__ == '__main__':
    main()

