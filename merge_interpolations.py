#!/usr/bin/env python3
"""
Merge greedy and interpolated training samples into a single CSV.

This script combines:
1. sft_training_samples_multi_greedy_parallel.csv (baseline greedy samples)
2. sft_training_samples_interpolated.csv (interpolated samples)

With optional filtering by number of questions.
"""

import pandas as pd
from pathlib import Path

########################################################
# CONFIGURATION - Edit these variables
########################################################

# Input CSV paths
GREEDY_CSV_PATH = './data/sft_training_samples_multi_greedy_parallel.csv'
INTERPOLATED_CSV_PATH = './data/sft_training_samples_interpolated.csv'

# Output CSV path
OUTPUT_CSV_PATH = './data/sft_training_samples_merged.csv'

# Number of questions to include (None = all questions)
NUM_QUESTIONS = 1  # Set to 10, 50, 100, etc. to limit, or None for all

# Print detailed statistics
VERBOSE = True

########################################################


def merge_training_samples(
    greedy_csv_path: str,
    interpolated_csv_path: str,
    output_csv_path: str,
    num_questions: int = None,
    verbose: bool = True
):
    """
    Merge greedy and interpolated training samples.
    
    Args:
        greedy_csv_path: Path to greedy samples CSV
        interpolated_csv_path: Path to interpolated samples CSV
        output_csv_path: Path to save merged CSV
        num_questions: Number of questions to include (None = all)
        verbose: Print detailed statistics
    """
    
    print(f"{'='*80}")
    print("🔄 MERGING TRAINING SAMPLES")
    print(f"{'='*80}")
    
    # Load both CSVs
    print(f"\n📂 Loading greedy samples from: {greedy_csv_path}")
    greedy_df = pd.read_csv(greedy_csv_path)
    print(f"   Loaded {len(greedy_df)} greedy samples")
    
    print(f"\n📂 Loading interpolated samples from: {interpolated_csv_path}")
    interpolated_df = pd.read_csv(interpolated_csv_path)
    print(f"   Loaded {len(interpolated_df)} interpolated samples")
    
    # Filter by number of questions if specified
    if num_questions is not None:
        print(f"\n🔍 Filtering to first {num_questions} questions...")
        
        # Get unique question IDs from greedy (sorted)
        unique_qids = sorted(greedy_df['question_id'].unique())
        selected_qids = unique_qids[:num_questions]
        
        print(f"   Selected question_ids: {selected_qids[:10]}{'...' if len(selected_qids) > 10 else ''}")
        
        # Filter both dataframes
        greedy_df = greedy_df[greedy_df['question_id'].isin(selected_qids)]
        interpolated_df = interpolated_df[interpolated_df['question_id'].isin(selected_qids)]
        
        print(f"   After filtering:")
        print(f"     Greedy: {len(greedy_df)} samples")
        print(f"     Interpolated: {len(interpolated_df)} samples")
    
    # Combine the dataframes
    print(f"\n🔗 Combining dataframes...")
    merged_df = pd.concat([greedy_df, interpolated_df], ignore_index=True)
    
    # Sort by question_id and position for consistency
    merged_df = merged_df.sort_values(['question_id', 'position'], ignore_index=True)
    
    print(f"   Total samples after merge: {len(merged_df)}")
    
    # Print statistics if verbose
    if verbose:
        print(f"\n📊 STATISTICS:")
        print(f"{'='*80}")
        
        # Samples per question
        samples_per_question = merged_df.groupby('question_id').size()
        print(f"   Samples per question:")
        print(f"     Mean: {samples_per_question.mean():.1f}")
        print(f"     Min: {samples_per_question.min()}")
        print(f"     Max: {samples_per_question.max()}")
        
        # Position distribution
        print(f"\n   Position distribution:")
        print(f"     Min position: {merged_df['position'].min()}")
        print(f"     Max position: {merged_df['position'].max()}")
        print(f"     Unique positions per question: {merged_df.groupby('question_id')['position'].nunique().mean():.1f}")
        
        # Block size distribution
        print(f"\n   Block size distribution:")
        print(f"     Mean: {merged_df['block_size'].mean():.2f}")
        print(f"     Min: {merged_df['block_size'].min()}")
        print(f"     Max: {merged_df['block_size'].max()}")
        
        # Question coverage
        unique_questions = merged_df['question_id'].nunique()
        print(f"\n   Question coverage:")
        print(f"     Unique questions: {unique_questions}")
        print(f"     Total samples: {len(merged_df)}")
        print(f"     Samples per question: {len(merged_df) / unique_questions:.1f}")
    
    # Save merged CSV
    print(f"\n💾 Saving merged samples to: {output_csv_path}")
    merged_df.to_csv(output_csv_path, index=False)
    print(f"   ✅ Saved {len(merged_df)} samples")
    
    print(f"\n{'='*80}")
    print("✅ MERGE COMPLETE")
    print(f"{'='*80}\n")
    
    return merged_df


def main():
    """Main execution using configuration variables."""
    
    # Check if input files exist
    if not Path(GREEDY_CSV_PATH).exists():
        print(f"❌ Error: Greedy CSV not found: {GREEDY_CSV_PATH}")
        return
    
    if not Path(INTERPOLATED_CSV_PATH).exists():
        print(f"❌ Error: Interpolated CSV not found: {INTERPOLATED_CSV_PATH}")
        return
    
    # Create output directory if needed
    output_dir = Path(OUTPUT_CSV_PATH).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge the samples
    merge_training_samples(
        greedy_csv_path=GREEDY_CSV_PATH,
        interpolated_csv_path=INTERPOLATED_CSV_PATH,
        output_csv_path=OUTPUT_CSV_PATH,
        num_questions=NUM_QUESTIONS,
        verbose=VERBOSE
    )


if __name__ == "__main__":
    main()
