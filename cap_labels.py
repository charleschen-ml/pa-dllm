#!/usr/bin/env python3
"""
Script to cap block_size labels in the training dataset.
Easily adjustable threshold for experimentation.
"""

import pandas as pd

########################################
# CONFIGURATION
########################################
CAP_TYPE = "block_size"  # Options: "block_size" or "block_size_rel"
THRESHOLD = 10  # Threshold value for capping
                # If CAP_TYPE="block_size": Try 8, 10, 12, 16, 20, 24, 32
                # If CAP_TYPE="block_size_rel": Try 0.5, 0.6, 0.7, 0.8, 0.9

########################################
# Main Script
########################################

# Load the CSV
csv_path = "./data/sft_training_samples_multi_greedy_parallel.csv"
print(f"📖 Loading {csv_path}...")
df = pd.read_csv(csv_path)

print(f"   Loaded {len(df)} samples")
print(f"   Current columns with block_size: {[c for c in df.columns if 'block_size' in c]}")
print(f"\n🎯 Capping mode: {CAP_TYPE}")
print(f"   Threshold: {THRESHOLD}")

if CAP_TYPE == "block_size":
    # ========================================
    # Mode 1: Cap block_size (integer tokens)
    # ========================================
    
    # Check if block_size_raw already exists (from previous run)
    if 'block_size_raw' in df.columns:
        print(f"\n🔄 Found existing block_size_raw, will re-cap from those values...")
        raw_values = df['block_size_raw']
    else:
        print(f"\n🔄 No block_size_raw found, using current block_size as raw values...")
        df.rename(columns={'block_size': 'block_size_raw'}, inplace=True)
        raw_values = df['block_size_raw']
    
    # Create new block_size column capped at THRESHOLD
    print(f"🔄 Creating capped block_size (max={THRESHOLD})...")
    df['block_size'] = raw_values.clip(upper=THRESHOLD)
    
    # Show statistics
    print(f"\n📊 Block size statistics:")
    print(f"   Raw block_size: min={raw_values.min()}, max={raw_values.max()}")
    print(f"   Capped block_size: min={df['block_size'].min()}, max={df['block_size'].max()}")
    num_capped = (raw_values > THRESHOLD).sum()
    print(f"   Samples capped at {THRESHOLD}: {num_capped} ({100*num_capped/len(df):.1f}%)")
    
    # Show distribution of capped values
    if num_capped > 0:
        print(f"   Distribution of values > {THRESHOLD}:")
        capped_distribution = raw_values[raw_values > THRESHOLD].value_counts().sort_index()
        for val, count in capped_distribution.items():
            print(f"      {val}: {count} samples")
    
    # Recalculate block_size_rel using capped block_size
    print(f"\n🔄 Recalculating block_size_rel...")
    # Infer base_block_length from original values: base_block_length = block_size_raw / block_size_rel
    # Handle division by zero
    df['base_block_length'] = raw_values / df['block_size_rel'].replace(0, 1)
    # Recalculate block_size_rel with capped block_size
    df['block_size_rel'] = df['block_size'] / df['base_block_length']
    
    print(f"\n📊 Block size relative statistics (after capping):")
    print(f"   Min: {df['block_size_rel'].min():.4f}")
    print(f"   Max: {df['block_size_rel'].max():.4f}")
    print(f"   Mean: {df['block_size_rel'].mean():.4f}")
    
    # Drop the temporary base_block_length column
    df.drop(columns=['base_block_length'], inplace=True)
    
    # Reorder columns to put block_size_raw next to block_size
    cols = list(df.columns)
    if 'block_size' in cols and 'block_size_raw' in cols:
        cols.remove('block_size_raw')
        block_size_idx = cols.index('block_size')
        cols.insert(block_size_idx + 1, 'block_size_raw')
        df = df[cols]
    
    print(f"\n📝 Updated column order: {list(df.columns)[:10]}...")

elif CAP_TYPE == "block_size_rel":
    # ========================================
    # Mode 2: Cap block_size_rel (fraction 0-1)
    # ========================================
    
    # Check if block_size_rel_raw already exists (from previous run)
    if 'block_size_rel_raw' in df.columns:
        print(f"\n🔄 Found existing block_size_rel_raw, will re-cap from those values...")
        raw_values = df['block_size_rel_raw']
    else:
        print(f"\n🔄 No block_size_rel_raw found, using current block_size_rel as raw values...")
        df.rename(columns={'block_size_rel': 'block_size_rel_raw'}, inplace=True)
        raw_values = df['block_size_rel_raw']
    
    # Create new block_size_rel column capped at THRESHOLD
    print(f"🔄 Creating capped block_size_rel (max={THRESHOLD})...")
    df['block_size_rel'] = raw_values.clip(upper=THRESHOLD)
    
    # Show statistics
    print(f"\n📊 Block size relative statistics:")
    print(f"   Raw block_size_rel: min={raw_values.min():.4f}, max={raw_values.max():.4f}")
    print(f"   Capped block_size_rel: min={df['block_size_rel'].min():.4f}, max={df['block_size_rel'].max():.4f}")
    num_capped = (raw_values > THRESHOLD).sum()
    print(f"   Samples capped at {THRESHOLD}: {num_capped} ({100*num_capped/len(df):.1f}%)")
    
    # Show distribution of capped values
    if num_capped > 0:
        print(f"   Distribution of values > {THRESHOLD} (binned):")
        capped_values = raw_values[raw_values > THRESHOLD]
        bins = [THRESHOLD + i*0.1 for i in range(int((capped_values.max() - THRESHOLD) / 0.1) + 2)]
        hist, bin_edges = pd.cut(capped_values, bins=bins, retbins=True, include_lowest=True)
        capped_distribution = hist.value_counts().sort_index()
        for interval, count in capped_distribution.items():
            print(f"      {interval}: {count} samples")
    
    # Recalculate block_size using capped block_size_rel
    print(f"\n🔄 Recalculating block_size...")
    # Infer base_block_length from original values
    # If we have block_size_raw, use it
    if 'block_size_raw' in df.columns:
        df['base_block_length'] = df['block_size_raw'] / raw_values.replace(0, 1)
    else:
        # Otherwise infer from current block_size
        df['base_block_length'] = df['block_size'] / raw_values.replace(0, 1)
        # Save original block_size as block_size_raw
        df['block_size_raw'] = df['block_size'].copy()
    
    # Recalculate block_size with capped block_size_rel
    df['block_size'] = (df['block_size_rel'] * df['base_block_length']).round().astype(int)
    
    print(f"\n📊 Block size statistics (after capping):")
    print(f"   Min: {df['block_size'].min()}")
    print(f"   Max: {df['block_size'].max()}")
    print(f"   Mean: {df['block_size'].mean():.2f}")
    
    # Drop the temporary base_block_length column
    df.drop(columns=['base_block_length'], inplace=True)
    
    # Reorder columns to put raw values next to capped values
    cols = list(df.columns)
    if 'block_size' in cols and 'block_size_raw' in cols:
        cols.remove('block_size_raw')
        block_size_idx = cols.index('block_size')
        cols.insert(block_size_idx + 1, 'block_size_raw')
    if 'block_size_rel' in cols and 'block_size_rel_raw' in cols:
        cols.remove('block_size_rel_raw')
        block_size_rel_idx = cols.index('block_size_rel')
        cols.insert(block_size_rel_idx + 1, 'block_size_rel_raw')
    df = df[cols]
    
    print(f"\n📝 Updated column order: {list(df.columns)[:12]}...")

else:
    raise ValueError(f"Invalid CAP_TYPE: {CAP_TYPE}. Must be 'block_size' or 'block_size_rel'.")

# Save back to the same file
print(f"\n💾 Saving to {csv_path}...")
df.to_csv(csv_path, index=False)

print(f"\n{'='*60}")
print(f"✅ Done! File updated successfully.")
print(f"{'='*60}")
print(f"   Total samples: {len(df)}")
print(f"   Columns: {len(df.columns)}")
print(f"   Capping mode: {CAP_TYPE}")
print(f"   {CAP_TYPE} capped at: {THRESHOLD}")
print(f"   Samples affected: {num_capped} ({100*num_capped/len(df):.1f}%)")
print(f"{'='*60}")

