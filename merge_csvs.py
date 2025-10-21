#!/usr/bin/env python3
"""
Merge multiple CSV files into one.
Merges sft_training_samples_multi_greedy_parallel_1k.csv through *_4k.csv
into sft_training_samples_multi_greedy.csv
"""

import pandas as pd
import glob
import os

def merge_csv_files():
    # Pattern to match the files in the data directory
    pattern = "data/sft_training_samples_multi_greedy_parallel_*.csv"
    
    # Find all matching files
    csv_files = sorted(glob.glob(pattern))
    
    if not csv_files:
        print(f"❌ No files found matching pattern: {pattern}")
        return
    
    print(f"📁 Found {len(csv_files)} files to merge:")
    for f in csv_files:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.2f} MB)")
    
    # Read and concatenate all CSVs
    print("\n🔄 Reading and merging CSVs with unique question IDs...")
    dfs = []
    total_rows = 0
    question_id_offset = 0
    
    for i, csv_file in enumerate(csv_files):
        df = pd.read_csv(csv_file)
        rows = len(df)
        total_rows += rows
        
        # Reassign question_id to be unique across all files
        if 'question_id' in df.columns:
            original_unique = df['question_id'].nunique()
            df['question_id'] = df['question_id'] + question_id_offset
            question_id_offset += original_unique
            print(f"  ✓ {csv_file}: {rows:,} rows, {original_unique} unique questions (offset by {question_id_offset - original_unique})")
        else:
            print(f"  ✓ {csv_file}: {rows:,} rows")
        
        dfs.append(df)
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save merged file
    output_file = "data/sft_training_samples_multi_greedy_parallel.csv"
    print(f"\n💾 Saving merged data to {output_file}...")
    merged_df.to_csv(output_file, index=False)
    
    output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Merged successfully!")
    print(f"   Total rows: {len(merged_df):,}")
    print(f"   Output file: {output_file} ({output_size_mb:.2f} MB)")
    
    # Show some stats
    print(f"\n📊 Quick stats:")
    print(f"   Unique questions: {merged_df['question_id'].nunique() if 'question_id' in merged_df.columns else 'N/A'}")
    print(f"   Columns: {list(merged_df.columns[:5])}{'...' if len(merged_df.columns) > 5 else ''}")

if __name__ == "__main__":
    merge_csv_files()

