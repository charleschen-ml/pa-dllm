#!/usr/bin/env python3
"""
Plot each feature vs block_size (label) for analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_features_vs_labels():
    """Plot each feature against block_size and save to output directory"""
    
    # Create output directory if it doesn't exist
    os.makedirs('./output', exist_ok=True)
    
    # Load the training data
    # csv_path = './data/sft_training_samples_greedy.csv' # single augmentation
    csv_path = './data/sft_training_samples_multi_greedy.csv' # multi augmentation
    if not os.path.exists(csv_path):
        print(f"❌ Data file not found: {csv_path}")
        return
    
    print(f"📊 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # manual experiment: filter by answer_found
    df = df[df['answer_found'] == False]
    
    print(f"✅ Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Define the features to plot (excluding non-numeric and identifier columns)
    feature_columns = [
        'confidence', 'entropy', 'position', 'position_relative',
        'conf_0', 'entropy_0', 'top1_margin', 'mean_confidence', 'mean_entropy',
        'conf_std', 'entropy_std', 'conf_1', 'top4_conf_min', 'next4_conf_min',
        'top8_conf_min', 'next8_conf_min'
    ]
    
    # Filter to only existing columns
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"📈 Plotting {len(available_features)} features: {available_features}")
    
    # Label column
    label_column = 'block_size'
    
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
    plt.xlabel('Correlation with block_size')
    plt.title('Feature Correlations with Block Size')
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
    print(f"\n🎯 Done! Generated {len(available_features)} feature plots + summary")

if __name__ == "__main__":
    plot_features_vs_labels()
