"""
Test Suite for Gesture Classification Pipeline

Tests:
1. Dataset loading and preprocessing
2. Model instantiation
3. Training loop
4. Inference pipeline
"""

import sys
from pathlib import Path

# Add models directory to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

# Import from models package
from models.gesture_dataset import GestureDataset
from models.gesture_classifier import create_model
from models.dataset_utils import get_data_loaders, analyze_dataset
from models.inference import GestureInference


def test_dataset():
    """Test dataset loading and preprocessing."""
    print("\n" + "="*60)
    print("TEST 1: Dataset Loading")
    print("="*60)
    
    dataset = GestureDataset("data/gestures", sequence_length=60)
    print(f"✓ Dataset created with {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("⚠️  No data found. This is expected on first run.")
        print("   Run main.py and record some gestures (press 1-5) before training.")
        return False
    
    # Test single sample
    sample, label = dataset[0]
    print(f"✓ Sample retrieved: shape {sample.shape}, label {label}")
    assert sample.shape == (60, 18), f"Expected (60, 18), got {sample.shape}"
    
    # Test preprocessing
    raw_segment = np.random.randn(45, 6, 3).astype(np.float32)
    tensor = dataset._process_sequence(raw_segment)
    assert tensor.shape == (60, 18), f"Expected (60, 18), got {tensor.shape}"
    print(f"✓ Preprocessing: Raw (45, 6, 3) → Processed (60, 18)")
    
    # Test class mapping
    for idx in range(len(dataset.VALID_LABELS)):
        label_name = dataset.get_label_name(idx)
        assert label_name in dataset.VALID_LABELS
    print(f"✓ Label mappings: {dataset.VALID_LABELS}")
    
    return True


def test_model():
    """Test model instantiation and forward pass."""
    print("\n" + "="*60)
    print("TEST 2: Model Instantiation & Forward Pass")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model = create_model(device=device)
    print(f"✓ Model created")
    
    # Test forward pass
    batch_size = 4
    seq_len = 60
    input_size = 18
    
    x = torch.randn(batch_size, seq_len, input_size).to(device)
    output = model(x)
    
    assert output.shape == (batch_size, 5), f"Expected {(batch_size, 5)}, got {output.shape}"
    print(f"✓ Forward pass: Input {x.shape} → Output {output.shape}")
    
    # Test save/load
    test_path = Path("models/test_model_temp.pth")
    model.save(str(test_path))
    assert test_path.exists()
    print(f"✓ Model saved")
    
    model2 = create_model(device=device)
    model2.load(str(test_path))
    print(f"✓ Model loaded")
    
    test_path.unlink()
    
    return True


def test_data_loaders():
    """Test data loaders."""
    print("\n" + "="*60)
    print("TEST 3: Data Loaders")
    print("="*60)
    
    dataset = GestureDataset("data/gestures", sequence_length=60)
    
    if len(dataset) < 10:
        print("⚠️  Dataset too small for proper train/val/test split")
        print("   Need at least 10 samples. Current:", len(dataset))
        return False
    
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset,
        batch_size=4,
        balance_train=True
    )
    
    print(f"✓ Loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test batch retrieval
    for batch_x, batch_y in train_loader:
        assert batch_x.shape[0] <= 4
        assert batch_x.shape[1:] == (60, 18)
        assert batch_y.shape[0] == batch_x.shape[0]
        print(f"✓ Batch retrieved: X {batch_x.shape}, Y {batch_y.shape}")
        break
    
    return True


def test_inference():
    """Test inference pipeline."""
    print("\n" + "="*60)
    print("TEST 4: Inference Pipeline")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Try to load real model
    model_path = Path("models/gesture_classifier_best.pth")
    if not model_path.exists():
        print("⚠️  No trained model found. Testing with random initialization...")
        
        # Create a model and save it for testing
        model = create_model(device=device)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(model_path))
    
    try:
        inf = GestureInference(
            model_path=str(model_path),
            sequence_length=60,
            device=device
        )
        print(f"✓ Inference wrapper loaded")
    except Exception as e:
        print(f"✗ Failed to load inference: {e}")
        return False
    
    # Test prediction
    test_segment = np.random.randn(40, 6, 3).astype(np.float32)
    label, confidence = inf.predict(test_segment)
    
    assert label in inf.dataset.VALID_LABELS
    assert 0.0 <= confidence <= 1.0
    print(f"✓ Prediction: {test_segment.shape} → '{label}' (conf: {confidence:.3f})")
    
    # Test batch prediction
    segments = [np.random.randn(np.random.randint(20, 80), 6, 3).astype(np.float32) 
                for _ in range(3)]
    results = inf.predict_batch(segments)
    assert len(results) == 3
    print(f"✓ Batch prediction: {len(results)} segments classified")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("GESTURE CLASSIFICATION PIPELINE - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Dataset Loading", test_dataset),
        ("Model Architecture", test_model),
        ("Data Loaders", test_data_loaders),
        ("Inference", test_inference)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "⚠️  SKIP/FAIL"
        print(f"{status:10} {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("1. Collect gesture data:")
        print("   - Run: python main.py")
        print("   - Press keys 1-5 to record different gestures")
        print("   - Perform gesture, then stop moving")
        print("")
        print("2. Train model:")
        print("   - Run: python models/train.py")
        print("")
        print("3. Evaluate model:")
        print("   - Run: python models/evaluate.py")
    else:
        print("\n⚠️  Some tests failed or were skipped.")
        print("\nTips:")
        print("- Make sure you have collected gesture data first")
        print("- Run: python main.py to record gestures")
        print("- Then run this test script again")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
