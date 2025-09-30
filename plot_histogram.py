#!/usr/bin/env python3
"""
Plot histogram of labels (block_size) from the training data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_label_histogram():
    """Plot histogram of labels (block_size) and save to output directory"""
    
    # Create output directory if it doesn't exist
    os.makedirs('./output', exist_ok=True)
    
    # Load the training data
    csv_path = './data/sft_training_samples_greedy.csv' # single augmentation
    # csv_path = './data/sft_training_samples_multi_greedy.csv' # multi augmentation
    if not os.path.exists(csv_path):
        print(f"❌ Data file not found: {csv_path}")
        return
    
    print(f"📊 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"✅ Loaded {len(df)} samples")
    
    # Filter by answer_found=False
    df = df[df['answer_found'] == False]
    print(f"🔍 Filtered to {len(df)} samples where answer_found=False")
    
    # Get the labels (block_size)
    labels = df['block_size']
    
    # Calculate statistics
    mean_val = labels.mean()
    median_val = labels.median()
    std_val = labels.std()
    min_val = labels.min()
    max_val = labels.max()
    
    # Print detailed statistics
    print(f"\n📊 Label Distribution Statistics:")
    print(f"Mean: {mean_val:.4f}")
    print(f"Median: {median_val:.4f}")
    print(f"Standard Deviation: {std_val:.4f}")
    print(f"Minimum: {min_val}")
    print(f"Maximum: {max_val}")
    print(f"Total Samples: {len(labels)}")
    
    # Print value counts
    print(f"\n🔢 Value Counts:")
    value_counts = labels.value_counts().sort_index()
    for value, count in value_counts.items():
        percentage = (count / len(labels)) * 100
        print(f"Block Size {value}: {count} samples ({percentage:.1f}%)")
    
    # Create a bar plot for value counts as well
    plt.figure(figsize=(12, 8))
    
    # Bar plot of value counts
    bars = plt.bar(value_counts.index, value_counts.values, alpha=0.7, color='lightcoral', edgecolor='black')
    
    # Add percentage labels on bars
    for bar, count in zip(bars, value_counts.values):
        height = bar.get_height()
        percentage = (count / len(labels)) * 100
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(value_counts.values),
                f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Block Size (Label)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Distribution of Labels (Block Size)\nTotal Samples: {len(labels)}', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set x-axis to show only integer ticks
    plt.xticks(value_counts.index)
    
    # Add statistics text box to bar plot
    stats_text = f'''Statistics:
Mean: {mean_val:.2f}
Median: {median_val:.2f}
Std Dev: {std_val:.2f}
Min: {min_val}
Max: {max_val}
Total Samples: {len(labels)}'''
    
    plt.text(0.7, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    # Save bar plot
    bar_output_path = './output/label_distribution_bars.png'
    plt.savefig(bar_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved bar plot: {bar_output_path}")
    print(f"\n🎯 Done! Generated bar plot of labels")

if __name__ == "__main__":
    plot_label_histogram()
