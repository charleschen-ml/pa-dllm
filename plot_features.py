#!/usr/bin/env python3
"""
Plot label histogram and feature correlations for analysis

This script generates:
1. Label distribution histogram (block_size)
2. Feature vs label scatter plots with correlations
3. Feature correlation summary and heatmap
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_label_histogram(csv_path):
    """Plot histogram of labels (block_size) and save to output directory"""
    
    # Create output directory if it doesn't exist
    os.makedirs('./output', exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"❌ Data file not found: {csv_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 PLOTTING LABEL HISTOGRAM")
    print(f"{'='*80}")
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
    print(f"🎯 Done! Generated bar plot of labels\n")


def plot_features_vs_labels(csv_path):
    """Plot each feature against block_size and save to output directory"""
    
    # Create output directory if it doesn't exist
    os.makedirs('./output', exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"❌ Data file not found: {csv_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"📈 PLOTTING FEATURE CORRELATIONS")
    print(f"{'='*80}")
    print(f"📊 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # manual experiment: filter by answer_found
    df = df[df['answer_found'] == False] # always do this
    # df = df[(df['position_relative'] > 0.7)]
    # df = df[(df['position_relative'] > 0.3) & (df['position_relative'] < 0.7)] # filter for middle positions
    
    print(f"✅ Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Define the features to plot (excluding non-numeric and identifier columns)
    # Removed entropy-based features (keeping only shannon entropy variants)
    feature_columns = [
        'position_relative', 'conf_0', 'shannon_entropy_0', 'top1_margin', 'mean_confidence',
        'shannon_mean_entropy', 'conf_std', 'shannon_entropy_std', 'conf_1', 'top4_conf_min', 'next4_conf_min',
        'top8_conf_min', 'next8_conf_min'
    ]
    
    # Filter to only existing columns
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"📈 Plotting {len(available_features)} features: {available_features}")
    
    # Label column
    # label_column = 'block_size'
    label_column = 'block_size_rel' # relative block size (block_size / remaining_length)
    
    if label_column not in df.columns:
        print(f"❌ Label column '{label_column}' not found in data")
        return
    
    # Plot each feature
    for i, feature in enumerate(available_features):
        print(f"Plotting {i+1}/{len(available_features)}: {feature}")
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Get data
        x = df[feature].dropna()
        y = df.loc[x.index, label_column]
        
        # Create scatter plot
        plt.scatter(x, y, alpha=0.6, s=30)
        
        # Add trend line if there are enough points
        if len(x) > 1:
            try:
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(x.min(), x.max(), 100)
                plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
                plt.legend()
            except:
                pass  # Skip trend line if fitting fails
        
        # Labels and title
        plt.xlabel(feature)
        plt.ylabel(label_column)
        plt.title(f'{feature} vs {label_column}')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
        plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}\nSamples: {len(x)}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        output_path = f'./output/{feature}_vs_{label_column}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {output_path}")
    
    # Create summary statistics
    print(f"\n📊 Creating summary statistics...")
    summary_stats = []
    
    for feature in available_features:
        x = df[feature].dropna()
        y = df.loc[x.index, label_column]
        
        if len(x) > 1:
            correlation = np.corrcoef(x, y)[0, 1]
            summary_stats.append({
                'feature': feature,
                'correlation': correlation,
                'samples': len(x),
                'feature_mean': x.mean(),
                'feature_std': x.std(),
                'label_mean': y.mean(),
                'label_std': y.std()
            })
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary_stats)
    summary_df = summary_df.sort_values('correlation', key=abs, ascending=False)
    summary_path = './output/feature_correlation_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print(f"✅ Saved summary: {summary_path}")
    print(f"\n🔍 Top correlated features:")
    print(summary_df[['feature', 'correlation', 'samples']].head(10).to_string(index=False))
    
    # Create a correlation heatmap
    plt.figure(figsize=(12, 8))
    correlations = [stats['correlation'] for stats in summary_stats]
    features = [stats['feature'] for stats in summary_stats]
    
    # Sort by absolute correlation
    sorted_indices = sorted(range(len(correlations)), key=lambda i: abs(correlations[i]), reverse=True)
    sorted_correlations = [correlations[i] for i in sorted_indices]
    sorted_features = [features[i] for i in sorted_indices]
    
    # Create bar plot
    colors = ['red' if corr < 0 else 'blue' for corr in sorted_correlations]
    bars = plt.barh(range(len(sorted_features)), sorted_correlations, color=colors, alpha=0.7)
    
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel(f'Correlation with {label_column}')
    plt.title(f'Feature Correlations with {label_column}')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add correlation values on bars
    for i, (bar, corr) in enumerate(zip(bars, sorted_correlations)):
        plt.text(corr + (0.01 if corr >= 0 else -0.01), i, f'{corr:.3f}', 
                va='center', ha='left' if corr >= 0 else 'right')
    
    plt.tight_layout()
    correlation_plot_path = './output/feature_correlations.png'
    plt.savefig(correlation_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved correlation plot: {correlation_plot_path}")
    print(f"🎯 Done! Generated {len(available_features)} feature plots + summary\n")


if __name__ == "__main__":
    # ========================================
    # CONFIGURATION
    # ========================================
    # Choose which data file to analyze
    # csv_path = './data/sft_training_samples_greedy.csv' # single augmentation
    # csv_path = './data/sft_training_samples_multi_greedy.csv' # multi augmentation
    csv_path = './data/sft_training_samples_multi_greedy_parallel.csv' # multi augmentation parallel
    
    print("="*80)
    print("🎨 FEATURE & LABEL VISUALIZATION")
    print("="*80)
    print(f"Data file: {csv_path}\n")
    
    # Plot label histogram
    plot_label_histogram(csv_path)
    
    # Plot feature correlations
    plot_features_vs_labels(csv_path)
    
    print("="*80)
    print("✅ ALL PLOTS COMPLETE!")
    print("="*80)
