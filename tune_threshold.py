"""
Tune the decision threshold for binary classification.

This script helps you find the optimal threshold to balance precision/recall.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# Load your trained model
MODEL_PATH = "./cache/block_size_scheduler.json"
DATA_PATH = "./data/sft_training_samples_multi_greedy_parallel.csv"

print("Loading model and data...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# Load test data (use the same split logic from train_scheduler.py)
df = pd.read_csv(DATA_PATH)

feature_cols = [
    'position_relative',
    'conf_0', 'conf_1', 'conf_2', 'conf_3', 'conf_4', 'conf_5', 'conf_6', 'conf_7', 'conf_8', 'conf_9',
    'shannon_entropy_0', 'shannon_entropy_1', 'shannon_entropy_2', 'shannon_entropy_3', 'shannon_entropy_4',
    'shannon_entropy_5', 'shannon_entropy_6', 'shannon_entropy_7', 'shannon_entropy_8', 'shannon_entropy_9',
    'top1_margin', 'mean_confidence', 'shannon_mean_entropy',
    'conf_std', 'shannon_entropy_std',
    'top4_conf_min', 'next4_conf_min', 'top8_conf_min', 'next8_conf_min'
]

X = df[feature_cols].fillna(df[feature_cols].median())
y = df['block_size'].apply(lambda bs: int(bs) - 1)  # Convert to 0-indexed

# Use test split (same as train_scheduler.py)
from sklearn.model_selection import train_test_split
if 'question_id' in df.columns:
    unique_questions = df['question_id'].unique()
    train_val_questions, test_questions = train_test_split(unique_questions, test_size=0.15, random_state=42)
    test_mask = df['question_id'].isin(test_questions)
    X_test = X[test_mask]
    y_test = y[test_mask]
else:
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print(f"Test set: {len(X_test)} samples")
print(f"  Class 0 (1 token): {(y_test == 0).sum()}")
print(f"  Class 1 (2 tokens): {(y_test == 1).sum()}")

# Get predicted probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (2 tokens)

# Try different thresholds
thresholds = np.arange(0.1, 1.0, 0.05)
results = []

print("\n" + "=" * 80)
print("THRESHOLD TUNING RESULTS")
print("=" * 80)
print(f"{'Threshold':<12} {'Accuracy':<10} {'Prec(1)':<10} {'Recall(1)':<10} {'F1(1)':<10} {'Prec(2)':<10}")
print("-" * 80)

for threshold in thresholds:
    # Predict class 1 (2 tokens) if probability > threshold
    y_pred = (y_pred_proba > threshold).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec_1 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    recall_1 = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_1 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
    prec_2 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    results.append({
        'threshold': threshold,
        'accuracy': acc,
        'precision_1': prec_1,
        'recall_1': recall_1,
        'f1_1': f1_1,
        'precision_2': prec_2
    })
    
    marker = "✅" if f1_1 > 0.55 else "  "
    print(f"{marker} {threshold:.2f}        {acc:.3f}      {prec_1:.3f}      {recall_1:.3f}      {f1_1:.3f}      {prec_2:.3f}")

# Find best threshold by F1 score
results_df = pd.DataFrame(results)
best_idx = results_df['f1_1'].idxmax()
best_threshold = results_df.loc[best_idx, 'threshold']
best_f1 = results_df.loc[best_idx, 'f1_1']

print("=" * 80)
print(f"\n🎯 Best threshold: {best_threshold:.2f} (F1 score for 1-token: {best_f1:.3f})")
print(f"   At this threshold:")
print(f"   - Accuracy: {results_df.loc[best_idx, 'accuracy']:.3f}")
print(f"   - Precision (1 tok): {results_df.loc[best_idx, 'precision_1']:.3f}")
print(f"   - Recall (1 tok): {results_df.loc[best_idx, 'recall_1']:.3f}")
print(f"   - F1 (1 tok): {results_df.loc[best_idx, 'f1_1']:.3f}")

# Plot precision-recall curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(results_df['threshold'], results_df['precision_1'], label='Precision (1 tok)', marker='o')
plt.plot(results_df['threshold'], results_df['recall_1'], label='Recall (1 tok)', marker='s')
plt.plot(results_df['threshold'], results_df['f1_1'], label='F1 (1 tok)', marker='^', linewidth=2)
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best ({best_threshold:.2f})')
plt.axvline(0.5, color='gray', linestyle=':', label='Default (0.5)')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Threshold vs Metrics (1-token class)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(results_df['threshold'], results_df['accuracy'], marker='o', color='green')
plt.axvline(best_threshold, color='red', linestyle='--', label=f'Best ({best_threshold:.2f})')
plt.axvline(0.5, color='gray', linestyle=':', label='Default (0.5)')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Threshold vs Overall Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./output/threshold_tuning.png', dpi=150)
print(f"\n📊 Plot saved to ./output/threshold_tuning.png")

print("\n" + "=" * 80)
print("USAGE IN INFERENCE:")
print("=" * 80)
print("```python")
print("# Load model")
print("model = xgb.XGBClassifier()")
print(f'model.load_model("{MODEL_PATH}")')
print("")
print("# Get probabilities")
print("prob_2_tokens = model.predict_proba([features])[0, 1]")
print(f"predicted_block_size = 2 if prob_2_tokens > {best_threshold:.2f} else 1")
print("```")
print("=" * 80)

