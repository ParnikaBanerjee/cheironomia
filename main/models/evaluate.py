"""
Evaluation Script for Gesture Classifier

Computes per-class metrics and confusion matrix on test set.

Usage:
    python models/evaluate.py [--data_dir DATA_DIR] [--model_path MODEL_PATH]
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add models directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from gesture_dataset import GestureDataset
from dataset_utils import get_data_loaders
from gesture_classifier import create_model


def evaluate(
    model,
    test_loader: DataLoader,
    dataset: GestureDataset,
    device: str = "cpu"
):
    """
    Evaluate model on test set and print metrics.
    
    Args:
        model: GestureClassifierLSTM instance
        test_loader: Test data loader
        dataset: GestureDataset instance (for label names)
        device: Device to evaluate on
    """
    model.eval()
    model.to(device)
    
    all_pred = []
    all_true = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            logits = model(sequences)
            predictions = torch.argmax(logits, dim=1)
            
            all_pred.extend(predictions.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
    
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    
    # Overall accuracy
    overall_acc = np.mean(all_pred == all_true)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nOverall Accuracy: {overall_acc:.4f}\n")
    
    # Per-class metrics
    print("Per-class Metrics:")
    print("-"*60)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*60)
    
    all_f1 = []
    
    for class_idx, label in enumerate(dataset.VALID_LABELS):
        # True positives, false positives, false negatives
        tp = np.sum((all_pred == class_idx) & (all_true == class_idx))
        fp = np.sum((all_pred == class_idx) & (all_true != class_idx))
        fn = np.sum((all_pred != class_idx) & (all_true == class_idx))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{label:<15} {precision:>11.4f} {recall:>11.4f} {f1:>11.4f}")
        all_f1.append(f1)
    
    print("-"*60)
    macro_f1 = np.mean(all_f1)
    print(f"{'Macro F1':<15} {macro_f1:>11.4f}\n")
    
    # Confusion matrix
    print("Confusion Matrix:")
    print("-"*60)
    
    # Helper function to format label names in header
    max_label_len = max(len(label) for label in dataset.VALID_LABELS)
    
    # Print header
    header = "True \\ Pred".ljust(max_label_len + 5)
    for label in dataset.VALID_LABELS:
        header += f"{label[:8]:<10}"
    print(header)
    print("-"*60)
    
    conf_matrix = np.zeros((len(dataset.VALID_LABELS), len(dataset.VALID_LABELS)), dtype=int)
    
    for true_idx in range(len(dataset.VALID_LABELS)):
        for pred_idx in range(len(dataset.VALID_LABELS)):
            count = np.sum((all_pred == pred_idx) & (all_true == true_idx))
            conf_matrix[true_idx, pred_idx] = count
    
    for true_idx, label in enumerate(dataset.VALID_LABELS):
        row_str = label.ljust(max_label_len + 5)
        for pred_idx in range(len(dataset.VALID_LABELS)):
            count = conf_matrix[true_idx, pred_idx]
            row_str += f"{count:<10d}"
        print(row_str)
    
    print("="*60 + "\n")
    
    # Visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(conf_matrix, cmap="Blues", aspect="auto")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(dataset.VALID_LABELS)))
        ax.set_yticks(np.arange(len(dataset.VALID_LABELS)))
        ax.set_xticklabels(dataset.VALID_LABELS, rotation=45, ha="right")
        ax.set_yticklabels(dataset.VALID_LABELS)
        
        # Add text annotations
        for i in range(len(dataset.VALID_LABELS)):
            for j in range(len(dataset.VALID_LABELS)):
                text = ax.text(j, i, conf_matrix[i, j],
                             ha="center", va="center", color="black", fontsize=12)
        
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("models/confusion_matrix.png", dpi=100)
        print("Confusion matrix plot saved to models/confusion_matrix.png")
        plt.close()
    except ImportError:
        print("matplotlib not installed. Skipping confusion matrix plot.")
    
    return {
        "accuracy": overall_acc,
        "macro_f1": macro_f1,
        "per_class_f1": all_f1,
        "confusion_matrix": conf_matrix
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate gesture classifier")
    parser.add_argument("--data_dir", type=str, default="data/gestures",
                        help="Path to gesture data directory")
    parser.add_argument("--model_path", type=str, default="models/gesture_classifier_best.pth",
                        help="Path to trained model checkpoint")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda). If None, auto-detect")
    
    args = parser.parse_args()
    
    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\nLoading dataset from {args.data_dir}...")
    dataset = GestureDataset(
        data_dir=args.data_dir,
        sequence_length=60,
        normalize_sequences=False
    )
    
    total_samples = len(dataset)
    if total_samples == 0:
        print("ERROR: No samples in dataset!")
        return
    
    # Create loaders (use all splits for evaluation)
    print(f"\nCreating data loaders...")
    _, _, test_loader = get_data_loaders(
        dataset,
        batch_size=32,
        stratify=True
    )
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    try:
        model = create_model(device=device)
        model.load(args.model_path)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {args.model_path}")
        print("\nTo train a model, run:")
        print("  python models/train.py")
        return
    
    # Evaluate
    evaluate(model, test_loader, dataset, device=device)


if __name__ == "__main__":
    main()
