#!/usr/bin/env python3
"""
Script to add conf_2...conf_9 and shannon_entropy_1...9 features.
DOES NOT modify any existing columns or labels.
"""

import pandas as pd
import numpy as np
import ast

########################################
# Load Data
########################################
csv_path = "./data/sft_training_samples_multi_greedy_parallel.csv"
print(f"📖 Loading {csv_path}...")
df = pd.read_csv(csv_path)

print(f"   Loaded {len(df)} samples")
print(f"   Current columns: {len(df.columns)}")

########################################
# Check if features already exist
########################################
new_conf_features = [f'conf_{i}' for i in range(2, 10)]
new_entropy_features = [f'shannon_entropy_{i}' for i in range(1, 10)]

existing_new = [f for f in new_conf_features + new_entropy_features if f in df.columns]
if existing_new:
    print(f"\n⚠️  Warning: Some new features already exist: {existing_new}")
    print(f"   They will be overwritten.")

########################################
# Helper Functions
########################################
def shannon_entropy(probs):
    """Calculate Shannon entropy from probability distribution."""
    probs = np.array(probs)
    # Remove zeros to avoid log(0)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))

def safe_parse_list(list_str):
    """Safely parse a string representation of a list."""
    try:
        return ast.literal_eval(list_str)
    except:
        return []

########################################
# Derive New Features ONLY
########################################
print(f"\n🔄 Deriving new features from full_confidence_list and full_entropy_list...")

# Parse full_confidence_list
print(f"   Parsing full_confidence_list...")
df['_parsed_conf'] = df['full_confidence_list'].apply(safe_parse_list)

# Extract conf_2 through conf_9 (8 additional features)
print(f"   Extracting conf_2 through conf_9...")
for i in range(2, 10):
    df[f'conf_{i}'] = df['_parsed_conf'].apply(
        lambda x: x[i] if len(x) > i else np.nan
    )

# Calculate shannon_entropy_1 through shannon_entropy_9 (9 additional features)
print(f"   Calculating shannon_entropy_1 through shannon_entropy_9...")

# Parse full_entropy_list if available
if 'full_entropy_list' in df.columns:
    print(f"      Using full_entropy_list...")
    df['_parsed_entropy'] = df['full_entropy_list'].apply(safe_parse_list)
    for i in range(1, 10):
        df[f'shannon_entropy_{i}'] = df['_parsed_entropy'].apply(
            lambda x: x[i] if len(x) > i else np.nan
        )
    df.drop(columns=['_parsed_entropy'], inplace=True)
else:
    print(f"      ⚠️  Warning: full_entropy_list not found, cannot compute shannon_entropy_1...9")

# Drop temporary columns
df.drop(columns=['_parsed_conf'], inplace=True)

# Show statistics on new features
print(f"\n📊 New feature statistics:")
print(f"   Added {len(new_conf_features)} confidence features: conf_2 ... conf_9")
print(f"   Added {len(new_entropy_features)} entropy features: shannon_entropy_1 ... shannon_entropy_9")
print(f"   Sample non-null counts:")
for feat in new_conf_features[:3]:  # Show first 3 as examples
    print(f"      {feat}: {df[feat].notna().sum()} / {len(df)}")
for feat in new_entropy_features[:3]:  # Show first 3 as examples
    print(f"      {feat}: {df[feat].notna().sum()} / {len(df)}")

########################################
# Save (keeping column order mostly intact)
########################################
print(f"\n💾 Saving to {csv_path}...")
df.to_csv(csv_path, index=False)

print(f"\n{'='*60}")
print(f"✅ Done! New features added successfully.")
print(f"{'='*60}")
print(f"   Total samples: {len(df)}")
print(f"   Total columns: {len(df.columns)} (was {len(df.columns) - len(new_conf_features) - len(new_entropy_features)})")
print(f"{'='*60}")

