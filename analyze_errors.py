"""
Analyze where the model is making mistakes and why.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and data
MODEL_PATH = "./cache/block_size_scheduler.json"
DATA_PATH = "./data/sft_training_samples_multi_greedy_parallel.csv"

print("Loading model and data...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

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
y = df['block_size'].apply(lambda bs: int(bs) - 1)  # 0-indexed

# Get test split
if 'question_id' in df.columns:
    unique_questions = df['question_id'].unique()
    train_val_questions, test_questions = train_test_split(unique_questions, test_size=0.15, random_state=42)
    test_mask = df['question_id'].isin(test_questions)
    X_test = X[test_mask]
    y_test = y[test_mask]
    df_test = df[test_mask].copy()
else:
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    df_test = df.loc[X_test.index].copy()

print(f"Test set: {len(X_test)} samples")

# Get predictions (use optimal threshold from tuning)
OPTIMAL_THRESHOLD = 0.25  # From threshold tuning
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > OPTIMAL_THRESHOLD).astype(int)

# Add predictions to dataframe
df_test['y_true'] = y_test.values
df_test['y_pred'] = y_pred
df_test['prob_class1'] = y_pred_proba
df_test['correct'] = (y_test.values == y_pred)

# Analyze errors
print("\n" + "=" * 80)
print("ERROR ANALYSIS")
print("=" * 80)

# False Positives: Predicted 1 token, actually 2 tokens
fp = df_test[(df_test['y_pred'] == 0) & (df_test['y_true'] == 1)]
print(f"\n❌ FALSE POSITIVES (predicted 1 token, should be 2): {len(fp)} samples")
if len(fp) > 0:
    print(f"   Avg confidence: {(1 - fp['prob_class1']).mean():.2%}")
    print(f"   Feature means:")
    print(f"     position_relative: {fp['position_relative'].mean():.3f}")
    print(f"     conf_0: {fp['conf_0'].mean():.3f}")
    print(f"     shannon_entropy_0: {fp['shannon_entropy_0'].mean():.3f}")
    print(f"     mean_confidence: {fp['mean_confidence'].mean():.3f}")

# False Negatives: Predicted 2 tokens, actually 1 token
fn = df_test[(df_test['y_pred'] == 1) & (df_test['y_true'] == 0)]
print(f"\n❌ FALSE NEGATIVES (predicted 2 tokens, should be 1): {len(fn)} samples")
if len(fn) > 0:
    print(f"   Avg confidence: {fn['prob_class1'].mean():.2%}")
    print(f"   Feature means:")
    print(f"     position_relative: {fn['position_relative'].mean():.3f}")
    print(f"     conf_0: {fn['conf_0'].mean():.3f}")
    print(f"     shannon_entropy_0: {fn['shannon_entropy_0'].mean():.3f}")
    print(f"     mean_confidence: {fn['mean_confidence'].mean():.3f}")

# True Positives: Correctly predicted 1 token
tp = df_test[(df_test['y_pred'] == 0) & (df_test['y_true'] == 0)]
print(f"\n✅ TRUE POSITIVES (correctly predicted 1 token): {len(tp)} samples")
if len(tp) > 0:
    print(f"   Avg confidence: {(1 - tp['prob_class1']).mean():.2%}")
    print(f"   Feature means:")
    print(f"     position_relative: {tp['position_relative'].mean():.3f}")
    print(f"     conf_0: {tp['conf_0'].mean():.3f}")
    print(f"     shannon_entropy_0: {tp['shannon_entropy_0'].mean():.3f}")
    print(f"     mean_confidence: {tp['mean_confidence'].mean():.3f}")

# True Negatives: Correctly predicted 2 tokens
tn = df_test[(df_test['y_pred'] == 1) & (df_test['y_true'] == 1)]
print(f"\n✅ TRUE NEGATIVES (correctly predicted 2 tokens): {len(tn)} samples")
if len(tn) > 0:
    print(f"   Avg confidence: {tn['prob_class1'].mean():.2%}")
    print(f"   Feature means:")
    print(f"     position_relative: {tn['position_relative'].mean():.3f}")
    print(f"     conf_0: {tn['conf_0'].mean():.3f}")
    print(f"     shannon_entropy_0: {tn['shannon_entropy_0'].mean():.3f}")
    print(f"     mean_confidence: {tn['mean_confidence'].mean():.3f}")

# Feature distribution comparison
print("\n" + "=" * 80)
print("FEATURE DISTRIBUTION: 1-token vs 2-token (TRUE LABELS)")
print("=" * 80)

class_0 = df_test[df_test['y_true'] == 0]
class_1 = df_test[df_test['y_true'] == 1]

# Calculate Cohen's d for ALL features and sort by effect size
feature_stats = []

for feat in feature_cols:
    if feat in df_test.columns:
        mean_0 = class_0[feat].mean()
        mean_1 = class_1[feat].mean()
        std_0 = class_0[feat].std()
        std_1 = class_1[feat].std()
        diff = abs(mean_0 - mean_1)
        
        # Measure separation (Cohen's d effect size)
        pooled_std = np.sqrt((std_0**2 + std_1**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        
        feature_stats.append({
            'feature': feat,
            'mean_0': mean_0,
            'std_0': std_0,
            'mean_1': mean_1,
            'std_1': std_1,
            'cohens_d': cohens_d
        })

# Sort by Cohen's d (descending) to show strongest predictors first
feature_stats_sorted = sorted(feature_stats, key=lambda x: x['cohens_d'], reverse=True)

# Print all features
for stat in feature_stats_sorted:
    feat = stat['feature']
    mean_0 = stat['mean_0']
    std_0 = stat['std_0']
    mean_1 = stat['mean_1']
    std_1 = stat['std_1']
    cohens_d = stat['cohens_d']
    
    marker = "🔴" if cohens_d < 0.2 else "🟡" if cohens_d < 0.5 else "🟢"
    print(f"{marker} {feat:25s}: 1tok={mean_0:6.3f}±{std_0:.3f}  2tok={mean_1:6.3f}±{std_1:.3f}  (d={cohens_d:.2f})")

print("\n📊 Cohen's d interpretation:")
print("   🔴 d < 0.2: Negligible separation (feature not useful)")
print("   🟡 0.2 ≤ d < 0.5: Small separation (weak signal)")
print("   🟢 d ≥ 0.5: Medium+ separation (useful feature)")
print("\n📈 Features sorted by predictive power (Cohen's d, highest to lowest)")

# Summary statistics
strong_features = [f for f in feature_stats_sorted if f['cohens_d'] >= 0.5]
weak_features = [f for f in feature_stats_sorted if 0.2 <= f['cohens_d'] < 0.5]
negligible_features = [f for f in feature_stats_sorted if f['cohens_d'] < 0.2]

print(f"\n🟢 Strong predictors (d ≥ 0.5): {len(strong_features)} features")
print(f"🟡 Weak predictors (0.2 ≤ d < 0.5): {len(weak_features)} features")
print(f"🔴 Negligible predictors (d < 0.2): {len(negligible_features)} features")

# Plot feature distributions for top 12 features (by Cohen's d)
top_features = [stat['feature'] for stat in feature_stats_sorted[:12]]

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for idx, feat in enumerate(top_features):
    if feat in df_test.columns:
        axes[idx].hist(class_0[feat].dropna(), bins=30, alpha=0.6, label='1 token', density=True, color='red')
        axes[idx].hist(class_1[feat].dropna(), bins=30, alpha=0.6, label='2 tokens', density=True, color='blue')
        
        # Get Cohen's d for this feature
        cohens_d = feature_stats_sorted[idx]['cohens_d']
        
        axes[idx].set_xlabel(feat, fontsize=10)
        axes[idx].set_ylabel('Density', fontsize=10)
        axes[idx].legend(fontsize=9)
        axes[idx].set_title(f'{feat} (d={cohens_d:.2f})', fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./output/feature_distributions.png', dpi=150)
print(f"\n📊 Feature distribution plot saved to ./output/feature_distributions.png")
print(f"   Showing top 12 features by Cohen's d")

# Print some specific error examples
print("\n" + "=" * 80)
print("EXAMPLE FALSE POSITIVES (predicted 1, should be 2)")
print("=" * 80)
if len(fp) > 0:
    for i, (idx, row) in enumerate(fp.head(5).iterrows()):
        print(f"\nExample {i+1}:")
        print(f"  Confidence: {(1 - row['prob_class1']):.2%} for 1 token")
        print(f"  position_relative: {row['position_relative']:.3f}")
        print(f"  conf_0: {row['conf_0']:.3f}, shannon_entropy_0: {row['shannon_entropy_0']:.3f}")
        print(f"  mean_confidence: {row['mean_confidence']:.3f}")
        if 'question_id' in row:
            print(f"  question_id: {row['question_id']}")

print("\n" + "=" * 80)
print("EXAMPLE FALSE NEGATIVES (predicted 2, should be 1)")
print("=" * 80)
if len(fn) > 0:
    for i, (idx, row) in enumerate(fn.head(5).iterrows()):
        print(f"\nExample {i+1}:")
        print(f"  Confidence: {row['prob_class1']:.2%} for 2 tokens")
        print(f"  position_relative: {row['position_relative']:.3f}")
        print(f"  conf_0: {row['conf_0']:.3f}, shannon_entropy_0: {row['shannon_entropy_0']:.3f}")
        print(f"  mean_confidence: {row['mean_confidence']:.3f}")
        if 'question_id' in row:
            print(f"  question_id: {row['question_id']}")

print("\n" + "=" * 80)
print("✅ Analysis complete!")
print("=" * 80)

