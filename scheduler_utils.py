#!/usr/bin/env python3
"""
Shared utilities for training and evaluating scheduler models.

This module contains common functions used by both XGBoost (train_scheduler.py)
and neural network (train_scheduler_head_features.py) schedulers.

Functions:
- split_by_question: Split data by question_id to avoid data leakage
- evaluate_classifier: Comprehensive evaluation with train/val/test metrics
- print_metrics_table: Pretty-print metrics in consistent format
- plot_confusion_matrix: Visualize confusion matrix
- plot_pr_curve: Visualize PR curve for binary classification
- blocksize_to_class / class_to_blocksize: Convert between representations
"""

import numpy as np
import pandas as pd
import random
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns


def blocksize_to_class(block_size, num_classes):
    """
    Convert block_size (1, 2, 3, ...) to class index (0, 1, 2, ...).
    
    Args:
        block_size: Integer representing number of tokens
        num_classes: Total number of classes (e.g., 2 for binary, 4 for 1-4 tokens)
        
    Returns:
        Class index (0-indexed, so block_size=1 → class=0)
    """
    # Clip to valid range
    block_size = int(block_size)
    block_size = max(1, min(num_classes, block_size))
    return block_size - 1  # Convert to 0-indexed


def class_to_blocksize(class_idx):
    """
    Convert class index back to block_size.
    
    Args:
        class_idx: Integer class index (0, 1, 2, ...)
        
    Returns:
        block_size: Integer number of tokens (1, 2, 3, ...)
    """
    return int(class_idx) + 1  # Convert from 0-indexed to 1-indexed


def split_by_question(csv_path: str, val_split: float, test_split: float = None, 
                     num_questions: int = None, seed: int = 42):
    """
    Split samples by question_id to avoid data leakage.
    
    Groups samples by their question_id, then splits questions into train/val/test.
    All samples from a question go to either train, val, or test - never mixed.
    
    Args:
        csv_path: Path to CSV file with samples
        val_split: Fraction of questions for validation (e.g., 0.1 = 10%)
        test_split: Fraction of questions for test (e.g., 0.2 = 20%). If None, only train/val split.
        num_questions: Limit to first N questions (None = use all)
        seed: Random seed for reproducible splits
    
    Returns:
        If test_split is None: (train_indices, val_indices)
        If test_split is provided: (train_indices, val_indices, test_indices)
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Group samples by question_id
    question_to_samples = {}
    for idx, row in df.iterrows():
        question_id = row.get('question_id', idx)  # Fallback to index if no question_id
        
        if question_id not in question_to_samples:
            question_to_samples[question_id] = []
        question_to_samples[question_id].append(idx)
    
    # Get list of unique question IDs (sorted for consistency)
    question_ids = sorted(question_to_samples.keys())
    
    # Limit to first N questions if specified
    if num_questions is not None:
        question_ids = question_ids[:num_questions]
    
    print(f"\n📊 Question-Level Split:")
    print(f"   Total unique questions: {len(question_ids)}")
    print(f"   Total samples: {sum(len(question_to_samples[q]) for q in question_ids)}")
    
    # Shuffle questions with fixed seed
    rng = random.Random(seed)
    rng.shuffle(question_ids)
    
    # Split questions into train/val/test
    # Edge case: if we have only 1 question, use it for all splits (for overfitting tests)
    if len(question_ids) == 1:
        print(f"   ⚠️  Only 1 question - using same question for train/val/test (overfit test)")
        train_question_ids = question_ids
        val_question_ids = question_ids
        test_question_ids = question_ids if test_split is not None else []
    else:
        if test_split is not None:
            # Three-way split
            num_test_questions = max(1, int(len(question_ids) * test_split))
            num_val_questions = max(1, int(len(question_ids) * val_split))
            
            test_question_ids = question_ids[:num_test_questions]
            val_question_ids = question_ids[num_test_questions:num_test_questions + num_val_questions]
            train_question_ids = question_ids[num_test_questions + num_val_questions:]
        else:
            # Two-way split (train/val only)
            num_val_questions = max(1, int(len(question_ids) * val_split))
            val_question_ids = question_ids[:num_val_questions]
            train_question_ids = question_ids[num_val_questions:]
            test_question_ids = []
    
    # Get sample indices for train, val, and test
    train_indices = []
    val_indices = []
    test_indices = []
    
    for q_id in train_question_ids:
        train_indices.extend(question_to_samples[q_id])
    
    for q_id in val_question_ids:
        val_indices.extend(question_to_samples[q_id])
    
    for q_id in test_question_ids:
        test_indices.extend(question_to_samples[q_id])
    
    print(f"   Train: {len(train_question_ids)} questions ({len(train_indices)} samples)")
    print(f"   Val:   {len(val_question_ids)} questions ({len(val_indices)} samples)")
    if test_split is not None:
        print(f"   Test:  {len(test_question_ids)} questions ({len(test_indices)} samples)")
    print(f"   ✅ No data leakage - questions are fully separated")
    
    if test_split is not None:
        return train_indices, val_indices, test_indices
    else:
        return train_indices, val_indices


def evaluate_classifier(y_true, y_pred, y_pred_proba=None, num_classes=2, 
                       split_name="Test", decision_threshold=0.5):
    """
    Comprehensive evaluation of a classifier.
    
    Args:
        y_true: True class labels (0, 1, 2, ...)
        y_pred: Predicted class labels (0, 1, 2, ...)
        y_pred_proba: Predicted probabilities [n_samples, num_classes] (optional, for AUC metrics)
        num_classes: Number of classes
        split_name: Name of the split ("Train", "Val", "Test")
        decision_threshold: Threshold for binary classification (only used for printing)
    
    Returns:
        dict with metrics:
        - accuracy: Overall accuracy
        - precision_per_class: List of precision scores per class
        - recall_per_class: List of recall scores per class
        - f1_per_class: List of F1 scores per class
        - support_per_class: List of support counts per class
        - classification_report: Full sklearn report dict
        - confusion_matrix: Confusion matrix array
        - pr_auc: PR-AUC for binary classification (if num_classes==2 and y_pred_proba provided)
        - roc_auc: ROC-AUC for binary classification (if num_classes==2 and y_pred_proba provided)
    """
    # Basic accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Classification report (per-class metrics)
    class_labels = [f'{i} tok' + ('s' if i > 1 else '') for i in range(1, num_classes + 1)]
    # Explicitly specify labels to handle missing classes gracefully
    labels_list = list(range(num_classes))  # [0, 1, 2, 3] for 4 classes
    report_dict = classification_report(
        y_true, y_pred,
        labels=labels_list,
        target_names=class_labels,
        digits=3,
        output_dict=True,
        zero_division=0
    )
    
    # Extract per-class metrics
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    support_per_class = []
    
    for class_idx in range(num_classes):
        class_name = class_labels[class_idx]
        if class_name in report_dict:
            precision_per_class.append(report_dict[class_name]['precision'])
            recall_per_class.append(report_dict[class_name]['recall'])
            f1_per_class.append(report_dict[class_name]['f1-score'])
            support_per_class.append(report_dict[class_name]['support'])
        else:
            precision_per_class.append(0.0)
            recall_per_class.append(0.0)
            f1_per_class.append(0.0)
            support_per_class.append(0)
    
    # Confusion matrix (explicitly specify labels to get correct shape even with missing classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    
    # Compile results
    results = {
        'split_name': split_name,
        'accuracy': accuracy,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class,
        'classification_report': report_dict,
        'confusion_matrix': cm,
        'num_classes': num_classes,
    }
    
    # Binary classification specific metrics (PR-AUC, ROC-AUC)
    if num_classes == 2 and y_pred_proba is not None:
        # Ensure 2D shape (n, 2) for consistency
        if y_pred_proba.ndim == 1:
            y_pred_proba = np.column_stack([1 - y_pred_proba, y_pred_proba])
        
        # Get probabilities for the minority class (class 0 = 1 token)
        y_proba_minority = y_pred_proba[:, 0]
        
        # Calculate PR-AUC (focuses on minority class = class 0)
        pr_auc = average_precision_score(y_true, y_proba_minority, pos_label=0)
        
        # Calculate ROC-AUC
        # Since we want class 0 as positive, we create binary labels: class 0 → 1, class 1 → 0
        y_true_binary = (y_true == 0).astype(int)
        roc_auc = roc_auc_score(y_true_binary, y_proba_minority)
        
        # Calculate baseline (frequency of minority class)
        minority_freq = np.mean(y_true == 0)
        
        results['pr_auc'] = pr_auc
        results['roc_auc'] = roc_auc
        results['minority_freq'] = minority_freq
    
    return results


def print_metrics_summary(train_results, val_results, test_results, 
                         decision_threshold=0.5, model_name="Model"):
    """
    Print a comprehensive summary of metrics across train/val/test splits.
    
    Args:
        train_results: Results dict from evaluate_classifier for train set
        val_results: Results dict from evaluate_classifier for val set
        test_results: Results dict from evaluate_classifier for test set
        decision_threshold: Decision threshold (for binary classification info)
        model_name: Name of the model (e.g., "XGBoost", "MLP")
    """
    num_classes = train_results['num_classes']
    
    print(f"\n🎯 Decision threshold: {decision_threshold} (for predicting class 1 = 2 tokens)")
    print(f"Train Accuracy: {train_results['accuracy']:.3f} ({100*train_results['accuracy']:.1f}%)")
    print(f"Val Accuracy:   {val_results['accuracy']:.3f} ({100*val_results['accuracy']:.1f}%)  ← Used for early stopping")
    print(f"Test Accuracy:  {test_results['accuracy']:.3f} ({100*test_results['accuracy']:.1f}%)  ← Final unseen evaluation")
    print(f"Overfitting gap (train-val): {train_results['accuracy'] - val_results['accuracy']:.3f}")
    
    # Binary classification specific metrics
    if num_classes == 2 and 'pr_auc' in test_results:
        print(f"\n📈 Area Under Curve Metrics (Minority Class = 1 token = class 0):")
        print(f"Train PR-AUC:  {train_results['pr_auc']:.3f}  |  ROC-AUC: {train_results['roc_auc']:.3f}")
        print(f"Val PR-AUC:    {val_results['pr_auc']:.3f}  |  ROC-AUC: {val_results['roc_auc']:.3f}")
        print(f"Test PR-AUC:   {test_results['pr_auc']:.3f}  |  ROC-AUC: {test_results['roc_auc']:.3f}")
        print(f"\n💡 PR-AUC is more informative for imbalanced data ({100*test_results['minority_freq']:.1f}% minority class)")
        print(f"   Baseline (random classifier):")
        print(f"     - PR-AUC ≈ {test_results['minority_freq']:.3f} (= minority class frequency)")
        print(f"     - ROC-AUC = 0.500")
    
    # Detailed per-class report (test set)
    print("\n📊 Detailed Classification Report (Test Set):")
    report = test_results['classification_report']
    class_labels = [f'{i} tok' + ('s' if i > 1 else '') for i in range(1, num_classes + 1)]
    
    # Print header
    print(f"              precision    recall  f1-score   support")
    print()
    
    # Print per-class metrics
    for class_idx, class_name in enumerate(class_labels):
        if class_name in report:
            p = report[class_name]['precision']
            r = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            s = int(report[class_name]['support'])
            print(f"   {class_name:>9}      {p:.3f}     {r:.3f}     {f1:.3f}      {s:4d}")
    
    print()
    
    # Print aggregate metrics
    if 'accuracy' in report:
        print(f"    accuracy                          {report['accuracy']:.3f}      {int(report['macro avg']['support']):4d}")
    if 'macro avg' in report:
        print(f"   macro avg      {report['macro avg']['precision']:.3f}     {report['macro avg']['recall']:.3f}     {report['macro avg']['f1-score']:.3f}      {int(report['macro avg']['support']):4d}")
    if 'weighted avg' in report:
        print(f"weighted avg      {report['weighted avg']['precision']:.3f}     {report['weighted avg']['recall']:.3f}     {report['weighted avg']['f1-score']:.3f}      {int(report['weighted avg']['support']):4d}")


def plot_confusion_matrix(y_true, y_pred, num_classes, save_path="./output/scheduler_confusion_matrix.png"):
    """
    Plot confusion matrix for classification results.
    
    Args:
        y_true: True class labels
        y_pred: Predicted class labels
        num_classes: Number of classes
        save_path: Path to save the plot
    """
    # Explicitly specify labels to get correct shape even with missing classes
    labels_list = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    
    # Adjust figure size based on number of classes
    fig_size = (8, 6) if num_classes <= 5 else (10, 8)
    plt.figure(figsize=fig_size)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Scheduler Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    
    # Generate class labels based on number of classes
    class_labels = [f'{i} tok' + ('s' if i > 1 else '') for i in range(1, num_classes + 1)]
    plt.gca().set_xticklabels(class_labels)
    plt.gca().set_yticklabels(class_labels)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix saved to {save_path}")


def plot_pr_curve(y_true, y_proba, save_path="./output/scheduler_pr_curve.png"):
    """
    Plot Precision-Recall curve for binary classification.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities for class 0 (minority class = 1 token)
        save_path: Path to save the plot
    """
    # Calculate precision-recall curve for class 0 (minority class)
    # pos_label=0 tells sklearn that class 0 is our target positive class
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba, pos_label=0)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_true, y_proba, pos_label=0)
    
    # Calculate baseline (random classifier) = frequency of class 0
    baseline = np.mean(y_true == 0)
    
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, linewidth=2.5, label=f'Model (AP = {avg_precision:.3f})', color='#2E86DE')
    plt.axhline(y=baseline, color='#EE5A6F', linestyle='--', linewidth=2, 
                label=f'Random Baseline = {baseline:.3f}')
    
    # Add annotation about what "good" looks like
    plt.text(0.5, 0.95, f'Minority class frequency: {baseline:.1%}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.xlabel('Recall (Sensitivity) - What % of 1-token cases we catch', fontsize=11)
    plt.ylabel('Precision (PPV) - When we predict 1-token, how often correct?', fontsize=11)
    plt.title(f'Precision-Recall Curve (Minority Class: 1 token)\nAP = {avg_precision:.3f} | Baseline = {baseline:.3f}', 
              fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"📈 Precision-Recall curve saved to {save_path}")


if __name__ == "__main__":
    # Simple test
    print("scheduler_utils.py - Shared utilities for scheduler training")
    print("Import this module in train_scheduler.py and train_scheduler_head_features.py")

