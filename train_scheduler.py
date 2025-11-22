"""
Train XGBoost scheduler to predict optimal block_size for fast inference.

This script trains a classifier to predict how many tokens (1 or 2)
should be decoded together, eliminating the need for expensive greedy sweeps.

Binary Classification Mode (default):
  Class 0: 1 token
  Class 1: 2 tokens

Multi-class Mode (set NUM_CLASSES > 2):
  Class 0: 1 token
  Class 1: 2 tokens
  Class 2: 3 tokens
  ... (up to NUM_CLASSES)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import os


def blocksize_to_class(block_size, num_classes):
    """
    Convert block_size (1, 2, 3, ...) to class index (0, 1, 2, ...).
    
    Args:
        block_size: Integer representing number of tokens
        num_classes: Total number of classes (e.g., 2 for binary, 5 for 1-5 tokens)
        
    Returns:
        Class index (0-indexed, so block_size=1 → class=0)
    """
    # Clip to valid range
    block_size = int(block_size)
    block_size = max(1, min(num_classes, block_size))
    return block_size - 1  # Convert to 0-indexed


def class_to_blocksize(class_idx):
    """
    Convert class index back to block_size.
    
    Args:
        class_idx: Integer class index (0, 1, 2, ...)
        
    Returns:
        block_size: Integer number of tokens (1, 2, 3, ...)
    """
    return int(class_idx) + 1  # Convert from 0-indexed to 1-indexed


def plot_pr_curve(y_true, y_proba, save_path="./output/scheduler_pr_curve.png"):
    """
    Plot Precision-Recall curve for binary classification.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for class 0 (minority class = 1 token)
        save_path: Path to save the plot
    """
    # Calculate precision-recall curve for class 0 (minority class)
    # pos_label=0 tells sklearn that class 0 is our target positive class
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba, pos_label=0)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_true, y_proba, pos_label=0)
    
    # Calculate baseline (random classifier) = frequency of class 0
    baseline = np.mean(y_true == 0)
    
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, linewidth=2.5, label=f'Model (AP = {avg_precision:.3f})', color='#2E86DE')
    plt.axhline(y=baseline, color='#EE5A6F', linestyle='--', linewidth=2, 
                label=f'Random Baseline = {baseline:.3f}')
    
    # Add annotation about what "good" looks like
    plt.text(0.5, 0.95, f'Minority class frequency: {baseline:.1%}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.xlabel('Recall (Sensitivity) - What % of 1-token cases we catch', fontsize=11)
    plt.ylabel('Precision (PPV) - When we predict 1-token, how often correct?', fontsize=11)
    plt.title(f'Precision-Recall Curve (Minority Class: 1 token)\nAP = {avg_precision:.3f} | Baseline = {baseline:.3f}', 
              fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📈 Precision-Recall curve saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, num_classes, save_path="./output/scheduler_confusion_matrix.png"):
    """Plot confusion matrix for classification results."""
    cm = confusion_matrix(y_true, y_pred)
    
    # Adjust figure size based on number of classes
    fig_size = (8, 6) if num_classes <= 5 else (10, 8)
    plt.figure(figsize=fig_size)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Scheduler Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    # Generate class labels based on number of classes
    class_labels = [f'{i} tok' + ('s' if i > 1 else '') for i in range(1, num_classes + 1)]
    plt.gca().set_xticklabels(class_labels)
    plt.gca().set_yticklabels(class_labels)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Confusion matrix saved to {save_path}")


def plot_feature_importance(model, feature_names, save_path="./output/scheduler_feature_importance.png"):
    """Plot feature importance from trained model."""
    importance = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importance)[::-1][:15]  # Top 15 features
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importance[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Most Important Features for Block Size Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Feature importance saved to {save_path}")


def main(use_wandb=True):
    print("=" * 80)
    print("TRAINING XGBOOST SCHEDULER FOR BLOCK SIZE PREDICTION")
    print("=" * 80)
    
    # ========================================
    # CONFIGURATION
    # ========================================
    USE_REGRESSION = False  # Set to True for regression, False for classification
    NUM_CLASSES = 2  # Number of token classes (2 for binary: 1 vs 2 tokens, 5 for 1-5 tokens, etc.)
    USE_CLASS_WEIGHTS = True  # Set to False to disable sample reweighting (sklearn mechanism)
    SCALE_POS_WEIGHT = 0  # XGBoost binary classification only: boost minority class
                             # None = auto-compute from class ratio (recommended)
                             # >0 = manual weight (e.g., 10.0 for 10x boost)
                             # 0 = disable (no boost)
                             # Note: Can be used WITH or WITHOUT USE_CLASS_WEIGHTS
    DECISION_THRESHOLD = 0.5  # Binary classification: threshold for predicting class 1 (2 tokens)
                              # Default: 0.5 (balanced)
                              # Lower (e.g., 0.3): predict 1-token more often → higher recall for class 0
                              # Higher (e.g., 0.7): predict 2-tokens more often → higher recall for class 1
    FILTER_BY_ANSWER_FOUND = False  # Set to True to filter out answer_found==True samples
    
    # ========================================
    # HYPERPARAMETERS - TUNE THESE!
    # ========================================
    HYPERPARAMS = {
        # Tree structure
        'n_estimators': 1000,      # Number of trees (more = better fit, slower). Try: 50, 100, 200, 500
        'max_depth': 6,           # Tree depth (lower = less overfitting). Try: 3, 4, 6, 9
        
        # Learning
        'learning_rate': 0.05,    # Learning rate (lower = slower, more careful). Try: 0.01, 0.05, 0.1, 0.3
        
        # Regularization (prevents overfitting) - uncomment to use
        'subsample': 0.8,         # Fraction of samples per tree
        'colsample_bytree': 0.8,  # Fraction of features per tree
        # 'min_child_weight': 5,    # Minimum samples in leaf (higher = smoother)
        # 'gamma': 1,               # Minimum loss reduction to split
        # 'reg_alpha': 0.5,           # L1 regularization
        # 'reg_lambda': 2,          # L2 regularization
        
        # Fixed parameters
        'random_state': 42,
    }
    
    # Update hyperparameters based on mode
    if USE_REGRESSION:
        HYPERPARAMS['objective'] = 'reg:squarederror'
        HYPERPARAMS['eval_metric'] = 'rmse'
        print(f"🎯 MODE: Regression (predicting continuous block_size_rel)")
    else:
        if NUM_CLASSES == 2:
            HYPERPARAMS['objective'] = 'binary:logistic'
            HYPERPARAMS['eval_metric'] = 'logloss'
            print(f"🎯 MODE: Binary Classification (1 vs 2 tokens)")
        else:
            HYPERPARAMS['objective'] = 'multi:softprob'
            HYPERPARAMS['num_class'] = NUM_CLASSES
            HYPERPARAMS['eval_metric'] = 'mlogloss'
            print(f"🎯 MODE: Multi-class Classification (1-{NUM_CLASSES} tokens)")
    
    # Note: scale_pos_weight will be added later after we compute class distribution
    
    # Data configuration
    DATA_PATH = "./data/sft_training_samples_multi_greedy_parallel.csv"
    # DATA_PATH = "./data/sft_training_samples_greedy.csv"
    MODEL_PATH = "./cache/block_size_scheduler.json"
    NUM_SAMPLES = None  # Set to N to use only first N samples (for overfitting experiments), or None for all samples
    TEST_SIZE = 0.15   # 15% for final test (completely unseen)
    VAL_SIZE = 0.15    # 15% for validation (used for early stopping)
                       # 70% for training
    
    # ========================================
    # Initialize wandb
    # ========================================
    if use_wandb:
        try:
            import wandb
            wandb_config = HYPERPARAMS.copy()
            wandb_config.update({
                'use_regression': USE_REGRESSION,
                'num_classes': NUM_CLASSES,
                'use_class_weights': USE_CLASS_WEIGHTS,
                'scale_pos_weight_config': SCALE_POS_WEIGHT,
                'decision_threshold': DECISION_THRESHOLD
            })
            wandb.init(
                project="pa-dllm-scheduler-training",
                name=f"xgboost_scheduler_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                config=wandb_config
            )
            print("📊 wandb initialized for training monitoring")
        except ImportError:
            print("⚠️  wandb not installed, skipping monitoring")
            use_wandb = False
    
    # Ensure output directory exists
    os.makedirs("./output", exist_ok=True)
    os.makedirs("./cache", exist_ok=True)
    
    # Load data
    print(f"\n📂 Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} samples")
    
    # Filter out samples where answer was already found (optional)
    if FILTER_BY_ANSWER_FOUND and 'answer_found' in df.columns:
        before_count = len(df)
        df = df[df['answer_found'] == False].reset_index(drop=True)
        after_count = len(df)
        print(f"🔍 Filtered out {before_count - after_count} samples where answer_found==True")
        print(f"   Remaining: {after_count} samples (before answer found)")
    
    # Filter to first N samples if NUM_SAMPLES is set (for overfitting experiments)
    if NUM_SAMPLES is not None:
        before_count = len(df)
        
        # Get first N unique question_ids if available, otherwise first N samples
        if 'question_id' in df.columns:
            unique_questions = df['question_id'].unique()[:NUM_SAMPLES]
            df = df[df['question_id'].isin(unique_questions)].reset_index(drop=True)
            print(f"🔬 Overfitting mode: Filtering to first {NUM_SAMPLES} question(s)")
            print(f"   Before: {before_count} samples")
            print(f"   After:  {len(df)} samples from {len(unique_questions)} question(s)")
        else:
            df = df.head(NUM_SAMPLES).reset_index(drop=True)
            print(f"🔬 Overfitting mode: Filtering to first {NUM_SAMPLES} sample(s)")
            print(f"   Before: {before_count} samples")
            print(f"   After:  {len(df)} samples")
    
    # Feature columns - Option 1: Minimal features (less missing values)
    # feature_cols = [
    #     'position_relative',
    #     # Confidence features (positions 0-1 only)
    #     'conf_0', 'conf_1',
    #     # Shannon entropy features (positions 0-1 only)
    #     'shannon_entropy_0', 'shannon_entropy_1',
    #     # Aggregate features
    #     'top1_margin', 'mean_confidence', 'shannon_mean_entropy',
    #     'conf_std', 'shannon_entropy_std',
    #     'top4_conf_min', 'next4_conf_min', 'top8_conf_min', 'next8_conf_min'
    # ]
    
    # Feature columns - Option 2: All features (30 features, some with missing values)
    # Uncomment below and comment out Option 1 above to use all features
    feature_cols = [
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
    
    # Monotonic constraints for XGBoost - Option 1: Minimal features
    # +1 = positive monotonicity (higher feature → higher block_size)
    # -1 = negative monotonicity (higher feature → lower block_size)
    #  0 = no constraint (let model learn freely)
    # monotonic_constraints = {
    #     'position_relative': 1,           # Later positions → larger blocks
    #     # Confidence features: higher confidence → larger blocks (+1)
    #     'conf_0': 1, 'conf_1': 1,
    #     # Shannon entropy features: higher entropy → smaller blocks (-1)
    #     'shannon_entropy_0': -1, 'shannon_entropy_1': -1,
    #     # Aggregate features
    #     'top1_margin': 1,                # Larger margin → larger blocks
    #     'mean_confidence': 1,            # Higher mean conf → larger blocks
    #     'shannon_mean_entropy': -1,      # Higher mean entropy → smaller blocks
    #     'conf_std': 0,                   # Let model learn freely (stddev of confidence)
    #     'shannon_entropy_std': 0,        # Let model learn freely (stddev of entropy)
    #     'top4_conf_min': 1,              # Higher min conf → larger blocks
    #     'next4_conf_min': 1,             # Higher min conf → larger blocks
    #     'top8_conf_min': 1,              # Higher min conf → larger blocks
    #     'next8_conf_min': 1              # Higher min conf → larger blocks
    # }
    
    # Monotonic constraints - Option 2: All features (uncomment if using Option 2 above)
    monotonic_constraints = {
        'position_relative': 1,           # Later positions → larger blocks
        # Confidence features: higher confidence → larger blocks (+1)
        'conf_0': 1, 'conf_1': 1, 'conf_2': 1, 'conf_3': 1, 'conf_4': 1,
        'conf_5': 1, 'conf_6': 1, 'conf_7': 1, 'conf_8': 1, 'conf_9': 1,
        # Shannon entropy features: higher entropy → smaller blocks (-1)
        'shannon_entropy_0': -1, 'shannon_entropy_1': -1, 'shannon_entropy_2': -1, 'shannon_entropy_3': -1,
        'shannon_entropy_4': -1, 'shannon_entropy_5': -1, 'shannon_entropy_6': -1, 'shannon_entropy_7': -1,
        'shannon_entropy_8': -1, 'shannon_entropy_9': -1,
        # Aggregate features
        'top1_margin': 1,                # Larger margin → larger blocks
        'mean_confidence': 1,            # Higher mean conf → larger blocks
        'shannon_mean_entropy': -1,      # Higher mean entropy → smaller blocks
        'conf_std': 0,                   # Let model learn freely (stddev of confidence)
        'shannon_entropy_std': 0,        # Let model learn freely (stddev of entropy)
        'top4_conf_min': 1,              # Higher min conf → larger blocks
        'next4_conf_min': 1,             # Higher min conf → larger blocks
        'top8_conf_min': 1,              # Higher min conf → larger blocks
        'next8_conf_min': 1              # Higher min conf → larger blocks
    }
    
    # Check for missing features
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Warning: Missing columns {missing_cols}")
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"📊 Using {len(feature_cols)} features")
    
    # Prepare features
    X = df[feature_cols]
    
    # Check for missing values and impute
    missing_count = X.isna().sum().sum()
    if missing_count > 0:
        missing_per_col = X.isna().sum()
        print(f"\n⚠️  Found {missing_count} missing values across {(missing_per_col > 0).sum()} features")
        print(f"   Imputation method: MEDIAN")
        for col in feature_cols:
            if missing_per_col[col] > 0:
                print(f"   - {col}: {missing_per_col[col]} missing")
        X = X.fillna(X.median())
    else:
        print(f"\n✅ No missing values found, no imputation needed")
    
    # Prepare target based on mode
    if USE_REGRESSION:
        print("\n🎯 Using continuous block_size_rel for regression...")
        y = df['block_size_rel']
        
        # Target distribution
        print(f"\n📊 Target distribution (block_size_rel):")
        print(f"  Mean:   {y.mean():.4f}")
        print(f"  Median: {y.median():.4f}")
        print(f"  Std:    {y.std():.4f}")
        print(f"  Min:    {y.min():.4f}")
        print(f"  Max:    {y.max():.4f}")
    else:
        print(f"\n🎯 Converting block_size to {NUM_CLASSES}-class classification...")
        y = df['block_size'].apply(lambda bs: blocksize_to_class(bs, NUM_CLASSES))
        
        # Class distribution
        print("\n📊 Class distribution:")
        for class_idx in range(NUM_CLASSES):
            block_size = class_to_blocksize(class_idx)
            count = (y == class_idx).sum()
            pct = 100 * count / len(y)
            token_str = f"{block_size} token" + ("s" if block_size > 1 else "")
            print(f"  Class {class_idx} ({token_str:>8}): {count:5d} samples ({pct:5.1f}%)")
        
        # Add scale_pos_weight for binary classification
        if NUM_CLASSES == 2:
            if SCALE_POS_WEIGHT is None:
                # Auto-compute: (count of class 1) / (count of class 0)
                count_class_0 = (y == 0).sum()
                count_class_1 = (y == 1).sum()
                scale_pos_weight_value = count_class_1 / count_class_0 if count_class_0 > 0 else 1.0
                HYPERPARAMS['scale_pos_weight'] = scale_pos_weight_value
                print(f"\n⚖️  Auto scale_pos_weight: {scale_pos_weight_value:.2f}")
                print(f"   (Boosts minority class to balance {count_class_1}:{count_class_0} ratio)")
            elif SCALE_POS_WEIGHT > 0:
                HYPERPARAMS['scale_pos_weight'] = SCALE_POS_WEIGHT
                print(f"\n⚖️  Manual scale_pos_weight: {SCALE_POS_WEIGHT:.2f}")
                print(f"   (Custom weight for minority class)")
            else:
                print(f"\n⚖️  scale_pos_weight disabled (set to 0)")
                print(f"   (No special weighting for minority class)")
    
    # Train/val/test split by question_id to prevent leakage
    train_pct = int((1 - TEST_SIZE - VAL_SIZE) * 100)
    val_pct = int(VAL_SIZE * 100)
    test_pct = int(TEST_SIZE * 100)
    
    # Check if we have enough data for proper splitting
    if 'question_id' in df.columns:
        unique_questions = df['question_id'].unique()
        num_unique_questions = len(unique_questions)
    else:
        num_unique_questions = len(df)
    
    # If too few samples/questions, skip split and use all data for training (overfitting mode)
    if num_unique_questions < 3:
        print(f"\n⚠️  Only {num_unique_questions} unique question(s) - skipping train/val/test split")
        print(f"   Using all {len(df)} samples for training (overfitting mode)")
        X_train = X
        y_train = y
        X_val = X  # Use same data for validation (no early stopping benefit)
        y_val = y
        X_test = X  # Use same data for test
        y_test = y
    else:
        print(f"\n🔀 Splitting data by question_id ({train_pct}% train, {val_pct}% val, {test_pct}% test)...")
        
        if 'question_id' in df.columns:
            # Split by question_id (all samples from same question go to same split)
            unique_questions = df['question_id'].unique()
            
            # First split: separate test set (completely unseen)
            train_val_questions, test_questions = train_test_split(
                unique_questions, test_size=TEST_SIZE, random_state=42
            )
            
            # Second split: split remaining into train and val
            # val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE) to get correct proportion
            val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
            train_questions, val_questions = train_test_split(
                train_val_questions, test_size=val_size_adjusted, random_state=42
            )
            
            # Filter dataframe by question_id
            train_mask = df['question_id'].isin(train_questions)
            val_mask = df['question_id'].isin(val_questions)
            test_mask = df['question_id'].isin(test_questions)
            
            X_train = X[train_mask]
            X_val = X[val_mask]
            X_test = X[test_mask]
            y_train = y[train_mask]
            y_val = y[val_mask]
            y_test = y[test_mask]
            
            print(f"  Train: {len(train_questions)} questions → {len(X_train)} samples")
            print(f"  Val:   {len(val_questions)} questions → {len(X_val)} samples (for early stopping)")
            print(f"  Test:  {len(test_questions)} questions → {len(X_test)} samples (completely unseen)")
            
            # Save question_id splits to files for inference
            from inference import generate_train_val_test_splits
            generate_train_val_test_splits(
                data_path=DATA_PATH,
                test_size=TEST_SIZE,
                val_size=VAL_SIZE,
                random_state=42,
                output_dir="./data/"
            )
        else:
            # Fallback: regular split if question_id not available
            print("⚠️  question_id not found, using sample-level split (may have leakage!)")
            # First split out test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=42
            )
            # Then split train_val into train and val
            val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42
            )
            print(f"  Train: {len(X_train)} samples")
            print(f"  Val:   {len(X_val)} samples")
            print(f"  Test:  {len(X_test)} samples")
    
    # Compute sample weights (only for classification)
    if not USE_REGRESSION and USE_CLASS_WEIGHTS:
        print("\n⚖️  Computing class weights for imbalanced data...")
        from sklearn.utils.class_weight import compute_sample_weight
        
        # Compute sample weights (inverse frequency weighting)
        sample_weights = compute_sample_weight('balanced', y_train)
        
        # Show weight distribution
        for class_idx in range(NUM_CLASSES):
            if (y_train == class_idx).sum() > 0:
                block_size = class_to_blocksize(class_idx)
                token_str = f"{block_size} token" + ("s" if block_size > 1 else "")
                avg_weight = sample_weights[y_train == class_idx].mean()
                print(f"  Class {class_idx} ({token_str:>8}): avg weight = {avg_weight:.2f}")
    else:
        if not USE_REGRESSION and not USE_CLASS_WEIGHTS:
            print("\n⚖️  Class reweighting disabled (USE_CLASS_WEIGHTS=False)")
            print("   Training with uniform sample weights (may favor majority class)")
        sample_weights = None
    
    # Train XGBoost Model
    model_type = "Regressor" if USE_REGRESSION else "Classifier"
    print(f"\n🚀 Training XGBoost {model_type}...")
    print(f"  Hyperparameters: n_estimators={HYPERPARAMS['n_estimators']}, "
          f"max_depth={HYPERPARAMS['max_depth']}, lr={HYPERPARAMS['learning_rate']}")
    if not USE_REGRESSION and NUM_CLASSES == 2 and 'scale_pos_weight' in HYPERPARAMS:
        print(f"  scale_pos_weight={HYPERPARAMS['scale_pos_weight']:.2f} (minority class boost)")
    
    # Add early stopping and monotonic constraints to hyperparameters
    model_params = HYPERPARAMS.copy()
    model_params['early_stopping_rounds'] = 200  # Stop if no improvement for 200 rounds
    
    # Convert monotonic constraints dict to tuple in feature order
    # XGBoost requires a tuple where position i corresponds to feature i
    monotone_constraints_tuple = tuple(monotonic_constraints[feat] for feat in feature_cols)
    model_params['monotone_constraints'] = monotone_constraints_tuple
    
    print(f"\n🔒 Monotonic constraints applied:")
    for feat in feature_cols:
        constraint = monotonic_constraints[feat]
        symbol = "↑" if constraint == 1 else "↓" if constraint == -1 else "○"
        constraint_str = "positive" if constraint == 1 else "negative" if constraint == -1 else "none"
        print(f"  {symbol} {feat}: {constraint_str}")
    
    if USE_REGRESSION:
        model = xgb.XGBRegressor(**model_params)
    else:
        model = xgb.XGBClassifier(**model_params)
    
    # Train model with early stopping (and sample weights for classification)
    # Use validation set for early stopping (NOT test set to prevent leakage)
    fit_kwargs = {
        'X': X_train,
        'y': y_train,
        'eval_set': [(X_val, y_val)],
        'verbose': False
    }
    if sample_weights is not None:
        fit_kwargs['sample_weight'] = sample_weights
    
    model.fit(**fit_kwargs)
    
    # Get the best iteration (before overfitting started)
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else HYPERPARAMS['n_estimators']
    print(f"  Best iteration: {best_iteration} (out of {HYPERPARAMS['n_estimators']} max)")
    
    print("✅ Training complete!")
    
    # Evaluate
    print("\n📈 EVALUATION RESULTS")
    print("=" * 80)
    
    if USE_REGRESSION:
        # Regression metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"Train RMSE: {train_rmse:.4f}  MAE: {train_mae:.4f}  R²: {train_r2:.4f}")
        print(f"Val RMSE:   {val_rmse:.4f}  MAE: {val_mae:.4f}  R²: {val_r2:.4f}  ← Used for early stopping")
        print(f"Test RMSE:  {test_rmse:.4f}  MAE: {test_mae:.4f}  R²: {test_r2:.4f}  ← Final unseen evaluation")
        print(f"Overfitting (train-val RMSE diff): {train_rmse - val_rmse:+.4f}")
    else:
        # Classification metrics
        # Get predictions
        if NUM_CLASSES == 2:
            # Binary classification: predict_proba returns shape (n, 2) or just (n,) in some versions
            y_train_pred_proba = model.predict_proba(X_train)
            y_val_pred_proba = model.predict_proba(X_val)
            y_test_pred_proba = model.predict_proba(X_test)
            
            # Ensure 2D shape (n, 2) for consistency
            if y_train_pred_proba.ndim == 1:
                y_train_pred_proba = np.column_stack([1 - y_train_pred_proba, y_train_pred_proba])
                y_val_pred_proba = np.column_stack([1 - y_val_pred_proba, y_val_pred_proba])
                y_test_pred_proba = np.column_stack([1 - y_test_pred_proba, y_test_pred_proba])
            
            y_train_pred = (y_train_pred_proba[:, 1] > DECISION_THRESHOLD).astype(int)
            y_val_pred = (y_val_pred_proba[:, 1] > DECISION_THRESHOLD).astype(int)
            y_test_pred = (y_test_pred_proba[:, 1] > DECISION_THRESHOLD).astype(int)
        else:
            # Multi-class: softprob returns probabilities, take argmax for class labels
            y_train_pred_proba = model.predict_proba(X_train)
            y_val_pred_proba = model.predict_proba(X_val)
            y_test_pred_proba = model.predict_proba(X_test)
            
            y_train_pred = np.argmax(y_train_pred_proba, axis=1)
            y_val_pred = np.argmax(y_val_pred_proba, axis=1)
            y_test_pred = np.argmax(y_test_pred_proba, axis=1)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print(f"\n🎯 Decision threshold: {DECISION_THRESHOLD} (for predicting class 1 = 2 tokens)")
        print(f"Train Accuracy: {train_acc:.3f} ({100*train_acc:.1f}%)")
        print(f"Val Accuracy:   {val_acc:.3f} ({100*val_acc:.1f}%)  ← Used for early stopping")
        print(f"Test Accuracy:  {test_acc:.3f} ({100*test_acc:.1f}%)  ← Final unseen evaluation")
        print(f"Overfitting gap (train-val): {train_acc - val_acc:.3f}")
        
        # Calculate PR-AUC and ROC-AUC for imbalanced classification
        # For binary classification, use class 0 (minority class) as positive
        if NUM_CLASSES == 2:
            # Get probabilities for the minority class (class 0 = 1 token)
            # Note: sklearn expects higher scores for positive class, so we use proba[:, 0]
            # and specify pos_label=0 to tell sklearn that class 0 is our target
            y_train_proba_minority = y_train_pred_proba[:, 0]
            y_val_proba_minority = y_val_pred_proba[:, 0]
            y_test_proba_minority = y_test_pred_proba[:, 0]
            
            # Calculate PR-AUC (focuses on minority class = class 0)
            train_pr_auc = average_precision_score(y_train, y_train_proba_minority, pos_label=0)
            val_pr_auc = average_precision_score(y_val, y_val_proba_minority, pos_label=0)
            test_pr_auc = average_precision_score(y_test, y_test_proba_minority, pos_label=0)
            
            # Calculate ROC-AUC (for comparison)
            # sklearn's roc_auc_score expects higher scores for positive class
            # Since we want class 0 as positive, we create binary labels: class 0 → 1, class 1 → 0
            y_train_binary = (y_train == 0).astype(int)
            y_val_binary = (y_val == 0).astype(int)
            y_test_binary = (y_test == 0).astype(int)
            
            train_roc_auc = roc_auc_score(y_train_binary, y_train_proba_minority)
            val_roc_auc = roc_auc_score(y_val_binary, y_val_proba_minority)
            test_roc_auc = roc_auc_score(y_test_binary, y_test_proba_minority)
            
            # Calculate true baseline (frequency of minority class = class 0)
            minority_freq_train = np.mean(y_train == 0)
            minority_freq_val = np.mean(y_val == 0)
            minority_freq_test = np.mean(y_test == 0)
            
            print(f"\n📈 Area Under Curve Metrics (Minority Class = 1 token = class 0):")
            print(f"Train PR-AUC:  {train_pr_auc:.3f}  |  ROC-AUC: {train_roc_auc:.3f}")
            print(f"Val PR-AUC:    {val_pr_auc:.3f}  |  ROC-AUC: {val_roc_auc:.3f}")
            print(f"Test PR-AUC:   {test_pr_auc:.3f}  |  ROC-AUC: {test_roc_auc:.3f}")
            print(f"\n💡 PR-AUC is more informative for imbalanced data ({100*minority_freq_test:.1f}% minority class)")
            print(f"   Baseline (random classifier):")
            print(f"     - PR-AUC ≈ {minority_freq_test:.3f} (= minority class frequency)")
            print(f"     - ROC-AUC = 0.500")
    
    # Log to wandb
    if use_wandb:
        if USE_REGRESSION:
            wandb.log({
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "test_rmse": test_rmse,
                "train_mae": train_mae,
                "val_mae": val_mae,
                "test_mae": test_mae,
                "train_r2": train_r2,
                "val_r2": val_r2,
                "test_r2": test_r2,
                "overfitting_rmse_diff": train_rmse - val_rmse,
                "num_train_samples": len(X_train),
                "num_val_samples": len(X_val),
                "num_test_samples": len(X_test)
            })
        else:
            wandb_dict = {
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "test_accuracy": test_acc,
                "overfitting_gap_train_val": train_acc - val_acc,
                "overfitting_gap_train_test": train_acc - test_acc,
                "num_train_samples": len(X_train),
                "num_val_samples": len(X_val),
                "num_test_samples": len(X_test)
            }
            # Add PR-AUC and ROC-AUC for binary classification
            if NUM_CLASSES == 2:
                wandb_dict.update({
                    "train_pr_auc": train_pr_auc,
                    "val_pr_auc": val_pr_auc,
                    "test_pr_auc": test_pr_auc,
                    "train_roc_auc": train_roc_auc,
                    "val_roc_auc": val_roc_auc,
                    "test_roc_auc": test_roc_auc,
                })
            wandb.log(wandb_dict)
    
    # Classification-specific reports
    if not USE_REGRESSION:
        print("\n📊 Detailed Classification Report (Test Set):")
        # Generate class labels
        class_labels = [f'{i} tok' + ('s' if i > 1 else '') for i in range(1, NUM_CLASSES + 1)]
        report = classification_report(
            y_test, y_test_pred,
            target_names=class_labels,
            digits=3,
            output_dict=True
        )
        print(classification_report(
            y_test, y_test_pred,
            target_names=class_labels,
            digits=3
        ))
        
        # Log per-class metrics to wandb
        if use_wandb:
            for class_idx, class_name in enumerate(class_labels):
                if class_name in report:
                    wandb.log({
                        f"precision_{class_name}": report[class_name]['precision'],
                        f"recall_{class_name}": report[class_name]['recall'],
                        f"f1_{class_name}": report[class_name]['f1-score'],
                    })
        
        # Plot PR curve for binary classification
        if NUM_CLASSES == 2:
            plot_pr_curve(y_test, y_test_proba_minority)
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_test_pred, NUM_CLASSES)
    
    # Plot feature importance
    plot_feature_importance(model, feature_cols)
    
    # Log plots to wandb
    if use_wandb:
        wandb_plots = {"feature_importance": wandb.Image("./output/scheduler_feature_importance.png")}
        if not USE_REGRESSION:
            wandb_plots["confusion_matrix"] = wandb.Image("./output/scheduler_confusion_matrix.png")
            if NUM_CLASSES == 2:
                wandb_plots["pr_curve"] = wandb.Image("./output/scheduler_pr_curve.png")
        wandb.log(wandb_plots)
    
    # Save model
    print(f"\n💾 Saving model to {MODEL_PATH}...")
    model.save_model(MODEL_PATH)
    print("✅ Model saved successfully!")
    
    # Finish wandb run
    if use_wandb:
        wandb.finish()
    
    # Example prediction
    print("\n🔍 EXAMPLE PREDICTIONS:")
    print("=" * 80)
    for i in range(min(5, len(X_test))):
        features = X_test.iloc[i:i+1]
        true_val = y_test.iloc[i]
        
        if USE_REGRESSION:
            pred_val = model.predict(features)[0]
            error = abs(true_val - pred_val)
            match = "✅" if error < 0.1 else "⚠️" if error < 0.2 else "❌"
            
            print(f"{match} Sample {i+1}:")
            print(f"   True: block_size_rel = {true_val:.4f}")
            print(f"   Pred: block_size_rel = {pred_val:.4f} (error: {error:.4f})")
        else:
            pred_proba = model.predict_proba(features)[0]
            if NUM_CLASSES == 2:
                # Binary: ensure we have 2 probabilities
                if pred_proba.ndim == 0 or len(pred_proba) == 1:
                    pred_proba = np.array([1 - pred_proba, pred_proba])
                pred_class = (pred_proba[1] > DECISION_THRESHOLD).astype(int)
            else:
                pred_class = np.argmax(pred_proba)
            
            true_blocksize = class_to_blocksize(true_val)
            pred_blocksize = class_to_blocksize(pred_class)
            
            match = "✅" if true_val == pred_class else "❌"
            pred_confidence = pred_proba[pred_class]
            
            print(f"{match} Sample {i+1}:")
            print(f"   True: class {true_val} ({true_blocksize} token{'s' if true_blocksize > 1 else ''})")
            print(f"   Pred: class {pred_class} ({pred_blocksize} token{'s' if pred_blocksize > 1 else ''}) (confidence: {pred_confidence:.2%})")
        
        print(f"   Features: conf_0={features['conf_0'].values[0]:.3f}, "
              f"shannon_entropy_0={features['shannon_entropy_0'].values[0]:.3f}, "
              f"pos_rel={features['position_relative'].values[0]:.3f}")
        print()
    
    print("=" * 80)
    print("✅ TRAINING COMPLETE!")
    if USE_REGRESSION:
        print(f"📈 Train RMSE: {train_rmse:.4f}  MAE: {train_mae:.4f}  R²: {train_r2:.4f}")
        print(f"📈 Val RMSE:   {val_rmse:.4f}  MAE: {val_mae:.4f}  R²: {val_r2:.4f} (used for early stopping)")
        print(f"📈 Test RMSE:  {test_rmse:.4f}  MAE: {test_mae:.4f}  R²: {test_r2:.4f} (final unseen evaluation)")
        print(f"   Overfitting (train-val RMSE diff): {train_rmse-val_rmse:+.4f}")
    else:
        print(f"📈 Train Accuracy: {train_acc:.1%}")
        print(f"📈 Val Accuracy:   {val_acc:.1%} (used for early stopping)")
        print(f"📈 Test Accuracy:  {test_acc:.1%} (final unseen evaluation)")
        print(f"   Overfitting gap (train-val): {train_acc-val_acc:+.1%}")
    print(f"📦 Model saved to: {MODEL_PATH}")
    print(f"📊 Plots saved to: ./output/")
    print("=" * 80)
    
    # Usage instructions
    print("\n📖 USAGE IN INFERENCE:")
    print("=" * 80)
    print("```python")
    print("import xgboost as xgb")
    print("import numpy as np")
    print("")
    print("# Load scheduler")
    if USE_REGRESSION:
        print(f"scheduler = xgb.XGBRegressor()")
    else:
        print(f"scheduler = xgb.XGBClassifier()")
    print(f'scheduler.load_model("{MODEL_PATH}")')
    print("")
    print("# At each position during inference:")
    if USE_REGRESSION:
        print("predicted_rel = scheduler.predict([features])[0]  # Direct prediction")
        print("remaining_tokens = max_length - curr_pos")
        print("predicted_block_size = max(1, int(predicted_rel * remaining_tokens))")
    else:
        if NUM_CLASSES == 2:
            print("# Binary classification:")
            print("prob_2_tokens = scheduler.predict_proba([features])[0, 1]")
            print("predicted_block_size = 2 if prob_2_tokens > 0.5 else 1")
        else:
            print(f"# Multi-class classification (1-{NUM_CLASSES} tokens):")
            print("class_idx = scheduler.predict([features])[0]")
            print("predicted_block_size = class_idx + 1  # Convert from 0-indexed to actual token count")
    print("```")
    print("=" * 80)
    

if __name__ == "__main__":
    main()

