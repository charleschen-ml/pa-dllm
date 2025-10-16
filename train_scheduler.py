"""
Train XGBoost scheduler to predict optimal block_size_rel for fast inference.

This script trains a classifier to predict which fraction of remaining tokens
can be decoded together, eliminating the need for expensive greedy sweeps.

Classes (power-of-2 bins):
  0: 1/32 (3.125% of remaining tokens)
  1: 1/16 (6.25%)
  2: 1/8  (12.5%)
  3: 1/4  (25%)
  4: 1/2  (50%)
  5: 1    (100% - all remaining tokens)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def rel_to_class(block_size_rel):
    """
    Convert block_size_rel (0-1) to nearest power-of-2 class.
    
    Args:
        block_size_rel: Float in [0, 1] representing fraction of remaining tokens
        
    Returns:
        Class index (0-5)
    """
    bins = [1/32, 1/16, 1/8, 1/4, 1/2, 1.0]
    class_idx = np.argmin([abs(block_size_rel - b) for b in bins])
    return class_idx


def class_to_rel(class_idx):
    """
    Convert class index back to block_size_rel.
    
    Args:
        class_idx: Integer in [0, 5]
        
    Returns:
        block_size_rel: Float representing fraction
    """
    bins = [1/32, 1/16, 1/8, 1/4, 1/2, 1.0]
    return bins[class_idx]


def plot_confusion_matrix(y_true, y_pred, save_path="./output/scheduler_confusion_matrix.png"):
    """Plot confusion matrix for classification results."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Scheduler Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    # Add class labels
    class_labels = ['1/32', '1/16', '1/8', '1/4', '1/2', '1']
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
    # HYPERPARAMETERS - TUNE THESE!
    # ========================================
    HYPERPARAMS = {
        # Tree structure
        'n_estimators': 3000,      # Number of trees (more = better fit, slower). Try: 50, 100, 200, 500
        'max_depth': 6,           # Tree depth (lower = less overfitting). Try: 3, 4, 6, 9
        
        # Learning
        'learning_rate': 0.05,    # Learning rate (lower = slower, more careful). Try: 0.01, 0.05, 0.1, 0.3
        
        # Regularization (prevents overfitting) - uncomment to use
        'subsample': 0.8,         # Fraction of samples per tree
        'colsample_bytree': 0.8,  # Fraction of features per tree
        'min_child_weight': 5,    # Minimum samples in leaf (higher = smoother)
        # 'gamma': 0,               # Minimum loss reduction to split
        'reg_alpha': 0.5,           # L1 regularization
        'reg_lambda': 2,          # L2 regularization
        
        # Fixed parameters
        'random_state': 42,
        'objective': 'multi:softprob',  # Output probabilities for smoother early stopping
        'num_class': 6,
        'eval_metric': 'mlogloss'
    }
    
    # Data configuration
    DATA_PATH = "./data/sft_training_samples_multi_greedy_parallel.csv"
    MODEL_PATH = "./cache/block_size_scheduler.json"
    TEST_SIZE = 0.15   # 15% for final test (completely unseen)
    VAL_SIZE = 0.15    # 15% for validation (used for early stopping)
                       # 70% for training
    
    # ========================================
    # Initialize wandb
    # ========================================
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="pa-dllm-scheduler-training",
                name=f"xgboost_scheduler_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
                config=HYPERPARAMS
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
    
    # Filter out samples where answer was already found
    if 'answer_found' in df.columns:
        before_count = len(df)
        df = df[df['answer_found'] == False].reset_index(drop=True)
        after_count = len(df)
        print(f"🔍 Filtered out {before_count - after_count} samples where answer_found==True")
        print(f"   Remaining: {after_count} samples (before answer found)")
    
    # Feature columns (17 features - removed 'confidence' and 'entropy' as they're redundant with 'conf_0' and 'entropy_0')
    feature_cols = [
        'position_relative',
        'conf_0', 'entropy_0', 'shannon_entropy_0', 'top1_margin',
        'mean_confidence', 'mean_entropy', 'shannon_mean_entropy',
        'conf_std', 'entropy_std', 'shannon_entropy_std', 'conf_1',
        'top4_conf_min', 'next4_conf_min', 'top8_conf_min', 'next8_conf_min'
    ]
    
    # Check for missing features
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Warning: Missing columns {missing_cols}")
        feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"📊 Using {len(feature_cols)} features")
    
    # Prepare features
    X = df[feature_cols].fillna(0)  # Fill NaN with 0
    
    # Prepare target: Convert block_size_rel to power-of-2 classes
    print("\n🎯 Converting block_size_rel to power-of-2 classes...")
    y = df['block_size_rel'].apply(rel_to_class)
    
    # Class distribution
    print("\n📊 Class distribution:")
    class_labels = ['1/32', '1/16', '1/8', '1/4', '1/2', '1']
    for class_idx in range(6):
        count = (y == class_idx).sum()
        pct = 100 * count / len(y)
        print(f"  Class {class_idx} ({class_labels[class_idx]:>4}): {count:5d} samples ({pct:5.1f}%)")
    
    # Train/val/test split by question_id to prevent leakage
    train_pct = int((1 - TEST_SIZE - VAL_SIZE) * 100)
    val_pct = int(VAL_SIZE * 100)
    test_pct = int(TEST_SIZE * 100)
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
    
    # Compute class weights to handle imbalance
    print("\n⚖️  Computing class weights for imbalanced data...")
    from sklearn.utils.class_weight import compute_sample_weight
    
    # Compute sample weights (inverse frequency weighting)
    sample_weights = compute_sample_weight('balanced', y_train)
    
    # Show weight distribution
    class_labels = ['1/32', '1/16', '1/8', '1/4', '1/2', '1']
    for class_idx in range(6):
        if (y_train == class_idx).sum() > 0:
            avg_weight = sample_weights[y_train == class_idx].mean()
            print(f"  Class {class_idx} ({class_labels[class_idx]:>4}): avg weight = {avg_weight:.2f}")
    
    # Train XGBoost Classifier
    print("\n🚀 Training XGBoost classifier...")
    print(f"  Hyperparameters: n_estimators={HYPERPARAMS['n_estimators']}, "
          f"max_depth={HYPERPARAMS['max_depth']}, lr={HYPERPARAMS['learning_rate']}")
    
    # Add early stopping to hyperparameters
    model_params = HYPERPARAMS.copy()
    model_params['early_stopping_rounds'] = 200  # Stop if no improvement for 50 rounds
    
    model = xgb.XGBClassifier(**model_params)
    
    # Train model with early stopping and sample weights
    # Use validation set for early stopping (NOT test set to prevent leakage)
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,  # Apply class weights
        eval_set=[(X_val, y_val)],  # Monitor validation set for early stopping
        verbose=False
    )
    
    # Get the best iteration (before overfitting started)
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else HYPERPARAMS['n_estimators']
    print(f"  Best iteration: {best_iteration} (out of {HYPERPARAMS['n_estimators']} max)")
    
    print("✅ Training complete!")
    
    # Evaluate
    print("\n📈 EVALUATION RESULTS")
    print("=" * 80)
    
    # Get predictions (softprob returns probabilities, take argmax for class labels)
    y_train_pred_proba = model.predict_proba(X_train)
    y_val_pred_proba = model.predict_proba(X_val)
    y_test_pred_proba = model.predict_proba(X_test)
    
    y_train_pred = np.argmax(y_train_pred_proba, axis=1)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)
    y_test_pred = np.argmax(y_test_pred_proba, axis=1)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"Train Accuracy: {train_acc:.3f} ({100*train_acc:.1f}%)")
    print(f"Val Accuracy:   {val_acc:.3f} ({100*val_acc:.1f}%)  ← Used for early stopping")
    print(f"Test Accuracy:  {test_acc:.3f} ({100*test_acc:.1f}%)  ← Final unseen evaluation")
    print(f"Overfitting gap (train-val): {train_acc - val_acc:.3f}")
    
    # Log to wandb
    if use_wandb:
        wandb.log({
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "test_accuracy": test_acc,
            "overfitting_gap_train_val": train_acc - val_acc,
            "overfitting_gap_train_test": train_acc - test_acc,
            "num_train_samples": len(X_train),
            "num_val_samples": len(X_val),
            "num_test_samples": len(X_test)
        })
    
    print("\n📊 Detailed Classification Report (Test Set):")
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
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_test_pred)
    
    # Plot feature importance
    plot_feature_importance(model, feature_cols)
    
    # Log plots to wandb
    if use_wandb:
        wandb.log({
            "confusion_matrix": wandb.Image("./output/scheduler_confusion_matrix.png"),
            "feature_importance": wandb.Image("./output/scheduler_feature_importance.png")
        })
    
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
        true_class = y_test.iloc[i]
        pred_proba = model.predict_proba(features)[0]  # Returns probabilities
        pred_class = np.argmax(pred_proba)  # Get class with highest probability
        
        true_rel = class_to_rel(true_class)
        pred_rel = class_to_rel(pred_class)
        
        match = "✅" if true_class == pred_class else "❌"
        pred_confidence = pred_proba[pred_class]  # Confidence in predicted class
        
        print(f"{match} Sample {i+1}:")
        print(f"   True: class {true_class} ({class_labels[true_class]}) = {true_rel:.4f}")
        print(f"   Pred: class {pred_class} ({class_labels[pred_class]}) = {pred_rel:.4f} (confidence: {pred_confidence:.2%})")
        print(f"   Features: conf_0={features['conf_0'].values[0]:.3f}, "
              f"entropy_0={features['entropy_0'].values[0]:.3f}, "
              f"pos_rel={features['position_relative'].values[0]:.3f}")
        print()
    
    print("=" * 80)
    print("✅ TRAINING COMPLETE!")
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
    print(f"scheduler = xgb.XGBClassifier()")
    print(f'scheduler.load_model("{MODEL_PATH}")')
    print("")
    print("# At each position during inference:")
    print("features = extract_features_at_position(model, tokenizer, prompt, curr_pos)")
    print("class_idx = scheduler.predict([features])[0]")
    print("")
    print("# Convert class to block_size")
    print("bins = [1/32, 1/16, 1/8, 1/4, 1/2, 1.0]")
    print("predicted_rel = bins[class_idx]")
    print("remaining_tokens = max_length - curr_pos")
    print("predicted_block_size = max(1, int(predicted_rel * remaining_tokens))")
    print("```")
    print("=" * 80)
    

if __name__ == "__main__":
    main()

