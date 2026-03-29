"""
Training Script for Gesture Classifier

Usage:
    python models/train.py [--data_dir DATA_DIR] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from gesture_dataset import GestureDataset
from dataset_utils import get_data_loaders, analyze_dataset
from gesture_classifier import create_model


class Trainer:
    """Trainer class for gesture classifier."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0
    ):
        """
        Args:
            model: GestureClassifierLSTM instance
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            device: Device to train on ("cpu" or "cuda")
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss function and optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }
        
        self.best_val_loss = float("inf")
        self.patience_counter = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            (avg_loss, avg_acc)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (sequences, labels) in enumerate(self.train_loader):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(sequences)
            loss = self.loss_fn(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item() * sequences.size(0)
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += sequences.size(0)
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate on validation set.
        
        Returns:
            (avg_loss, avg_acc)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for sequences, labels in self.val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(sequences)
                loss = self.loss_fn(logits, labels)
                
                total_loss += loss.item() * sequences.size(0)
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += sequences.size(0)
        
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def train(
        self,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_dir: str = "models"
    ):
        """
        Train the model for specified number of epochs.
        
        Args:
            epochs: Number of training epochs
            early_stopping_patience: Stop if validation loss doesn't improve for N epochs
            checkpoint_dir: Directory to save best model checkpoint
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(checkpoint_dir) / "gesture_classifier_best.pth"
        
        print(f"\n=== Training Start ===")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print()
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}", end="")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.model.save(str(checkpoint_path))
                print(" ✓ (saved)")
            else:
                self.patience_counter += 1
                print()
                
                if self.patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
        
        elapsed = time.time() - start_time
        print(f"\n=== Training Complete ===")
        print(f"Time elapsed: {elapsed:.1f}s")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best checkpoint: {checkpoint_path}")
    
    def plot_history(self, save_path: str = "models/training_history.png"):
        """Plot training history."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Skipping plot.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.history["train_loss"], label="Train Loss", marker="o")
        axes[0].plot(self.history["val_loss"], label="Val Loss", marker="o")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training History - Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(self.history["train_acc"], label="Train Acc", marker="o")
        axes[1].plot(self.history["val_acc"], label="Val Acc", marker="o")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training History - Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        print(f"Training history plot saved to {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train gesture classifier")
    parser.add_argument("--data_dir", type=str, default="data/gestures",
                        help="Path to gesture data directory")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--early_stopping_patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda). If None, auto-detect")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}\n")
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    dataset = GestureDataset(
        data_dir=args.data_dir,
        sequence_length=60,
        normalize_sequences=False
    )
    
    # Analyze dataset
    analyze_dataset(dataset)
    
    # Check if we have enough data
    total_samples = len(dataset)
    if total_samples < 50:
        print(f"\n⚠️  WARNING: Very small dataset ({total_samples} samples)")
        print("Please collect more gesture data before training a proper model.")
        print("\nUsage:")
        print("  1. Run main.py to record gestures")
        print("  2. Press keys 1-5 to record different gesture types")
        print("  3. Perform gesture, then stop moving to trigger segmentation")
        print("  4. Run this script again once you have ≥100 samples")
        return
    
    # Create data loaders
    print(f"\nCreating data loaders with batch_size={args.batch_size}...")
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset,
        batch_size=args.batch_size,
        balance_train=True,
        stratify=True
    )
    
    # Create model
    print(f"\nCreating model...")
    model = create_model(device=device)
    model.summary()
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate
    )
    
    trainer.train(
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir="models"
    )
    
    # Plot history
    trainer.plot_history()
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
