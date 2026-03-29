# Gesture Classification Training Guide

## Overview

This document walks you through the complete gesture classification pipeline for the Cheironomia Virtual Orchestra system.

**Components:**

- **Data Collection** → Gesture recording via webcam
- **Model Training** → LSTM classifier trained on collected data
- **Real-time Inference** → Live gesture classification during performance

---

## Phase 1: Installation & Setup

### 1.1 Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:

- `opencv-python` — Webcam capture
- `mediapipe` — Hand/pose landmark detection
- `numpy` — Numerical operations
- `torch` — Deep learning (LSTM model)
- `scikit-learn` — ML utilities (stratified split, metrics)

### 1.2 Verify Installation

```bash
python test_pipeline.py
```

You should see:

```
✓ PASS: Model Architecture
✓ PASS: Inference
...
```

---

## Phase 2: Data Collection

### 2.1 Record Gesture Data

Run the main data collection script:

```bash
python main.py
```

### 2.2 Recording Workflow

1. **Webcam opens** — Shows your skeleton and hands
2. **Press key to arm recording**:
   - `1` = Record next gesture as **START** (upward sweep)
   - `2` = Record next gesture as **STOP** (downward motion + freeze)
   - `3` = Record next gesture as **ACCENT** (downward jab)
   - `4` = Record next gesture as **CUT** (horizontal slice)
   - `5` = Record next gesture as **PATTERN_CHANGE** (circular motion)
3. **Perform the gesture** — Move your right arm
4. **Stop moving** — System detects end of motion and saves segment
5. **Repeat** — Record 20–30 samples per gesture type

### 2.3 Gesture Definitions

Ensure each gesture is **visually distinct**:

| Gesture            | Motion             | Visual Cue                           |
| ------------------ | ------------------ | ------------------------------------ |
| **START**          | Upward sweep       | Right arm rises with velocity        |
| **STOP**           | Downward + freeze  | Quick downward jab, then hold still  |
| **ACCENT**         | Short downward jab | Brief downward punch (not full stop) |
| **CUT**            | Horizontal slice   | Side-to-side cutting motion          |
| **PATTERN_CHANGE** | Circular           | Arm traces circle in front of body   |

### 2.4 Data Quality Tips

- **Variability**: Record gestures at **different speeds** and **amplitudes**
- **Multiple angles**: Perform from left, right, and center positions
- **Natural motion**: Avoid overly slow or robotic movements
- **Minimum samples**: Aim for **50–100 per class** (~250–500 total)

### 2.5 Data Storage

Recorded segments are saved to:

```
data/gestures/
├── start_1234567890.npy
├── start_1234567891.npy
├── stop_1234567892.npy
├── accident_1234567893.npy
└── ...
```

Each `.npy` file contains a normalized sequence:

- Shape: `(T, 6, 3)` where T=variable time steps
- `6` joints: left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist
- `3` coordinates: x, y, z (relative to body center)

---

## Phase 3: Training

### 3.1 Check Data

Verify you have enough samples:

```bash
python models/train.py --data_dir data/gestures
```

This will print dataset statistics. You should see:

```
Dataset loaded: 250 sequences
Label distribution:
  start: 50
  stop: 50
  accent: 50
  cut: 50
  pattern_change: 50
```

**Minimum requirement**: ≥20 samples per class, ideally ≥50

### 3.2 Start Training

```bash
python models/train.py \
    --data_dir data/gestures \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 0.001
```

**Parameters**:

- `--epochs`: Number of training iterations (default: 50)
- `--batch_size`: Samples per batch (default: 16)
- `--learning_rate`: Optimizer step size (default: 0.001)
- `--early_stopping_patience`: Stop if no improvement for N epochs (default: 10)

### 3.3 Training Progress

You'll see output like:

```
=== Training Start ===
Device: cuda
Epochs: 50
Early stopping patience: 10

Epoch   1/50 | Train Loss: 1.6021 | Train Acc: 0.2000 | Val Loss: 1.5123 | Val Acc: 0.2667 ✓ (saved)
Epoch   2/50 | Train Loss: 1.4562 | Train Acc: 0.4125 | Val Loss: 1.3456 | Val Acc: 0.5333 ✓ (saved)
Epoch   3/50 | Train Loss: 1.1234 | Train Acc: 0.6250 | Val Loss: 1.0123 | Val Acc: 0.7333 ✓ (saved)
...
Epoch  15/50 | Train Loss: 0.2345 | Train Acc: 0.9375 | Val Loss: 0.3456 | Val Acc: 0.8667
```

**What to look for**:

- ✓ Loss decreases each epoch
- ✓ Accuracy increases
- ✓ Best weights saved automatically

### 3.4 Training Stops When

- **Early stopping triggered**: Validation loss hasn't improved for 10 consecutive epochs
- **All epochs completed**: Reached max 50 epochs

**Duration**: ~5–10 minutes on CPU, ~1–2 minutes on GPU

### 3.5 Output

After training completes:

```
=== Training Complete ===
Time elapsed: 234.5s
Best validation loss: 0.3456
Best checkpoint: models/gesture_classifier_best.pth
```

Best model saved to: `models/gesture_classifier_best.pth`

Optional training history plot: `models/training_history.png`

---

## Phase 4: Evaluation

### 4.1 Test on Unseen Data

```bash
python models/evaluate.py \
    --data_dir data/gestures \
    --model_path models/gesture_classifier_best.pth
```

### 4.2 Evaluation Results

You'll see detailed per-class metrics:

```
============================================================
EVALUATION RESULTS
============================================================

Overall Accuracy: 0.8667

Per-class Metrics:
------------------------------------------------------------
Class               Precision    Recall      F1-Score
------------------------------------------------------------
start                  0.8500    0.8000      0.8235
stop                   0.9000    0.9000      0.9000
accent                 0.8000    0.8333      0.8163
cut                    0.8667    0.8667      0.8667
pattern_change         0.8500    0.8000      0.8235
------------------------------------------------------------
Macro F1                                     0.8660

Confusion Matrix:
...
```

**Interpretation**:

- **Accuracy > 80%**: Good model
- **Accuracy 70–80%**: Acceptable
- **Accuracy < 70%**: Need more or better training data

If accuracy is low:

- Collect more data (especially for low-performing classes)
- Ensure gestures are visually distinct
- Retrain the model

### 4.3 Confusion Matrix Visualization

Saved to: `models/confusion_matrix.png`

Shows which gestures are commonly confused with each other.

---

## Phase 5: Real-time Inference

### 5.1 Live Gesture Classification

Once trained, the system automatically classifies gestures in real-time:

```bash
python main.py
```

When you perform a gesture:

```
Tempo: 120.5 BPM | Intensity: 0.657
  → Gesture: START (confidence: 0.923)
```

**Output format**:

- `Tempo`: Beats per minute (extracted from right-arm oscillation)
- `Intensity`: Energy level [0, 1] (extracted from right-arm velocity)
- `Gesture`: Predicted gesture label
- `Confidence`: Prediction confidence [0, 1]

### 5.2 Confidence Threshold

Gestures with confidence < 0.3 are marked as "uncertain":

```
→ Gesture: uncertain (confidence: 0.2156)
```

You can adjust this in [main.py](main.py):

```python
gesture_classifier = GestureInference(..., confidence_threshold=0.3)
```

---

## Troubleshooting

### Problem: "No .npy files found in data/gestures"

**Solution**: You haven't recorded any gestures yet. Run `python main.py` and record at least 20 samples per gesture type.

### Problem: "Model file not found at models/gesture_classifier_best.pth"

**Solution**: You haven't trained a model yet. Run `python models/train.py`.

### Problem: Low accuracy on test set

**Possible causes**:

1. **Too little data**: Collect ≥50 samples per class
2. **Poor data quality**: Gestures aren't visually distinct enough
3. **Overfitting**: Collect more diverse variations
4. **Bad hyperparameters**: Try different learning rates or epochs

**Solutions**:

1. Record more data with higher variability
2. Ensure gestures are clearly different (not just speed variations)
3. Retrain with `--epochs 100` or lower learning rate

### Problem: Model predicts same gesture for everything

**Solution**: Your training data is likely imbalanced. The dataset utilities apply weighted sampling automatically, but check:

```bash
python -c "from models.gesture_dataset import GestureDataset; ds = GestureDataset(); print(ds.get_class_counts())"
```

If one class has >5x more samples than others, rebalance by recording fewer of the common class.

### Problem: Real-time inference is slow

**Solution**: Using GPU is much faster. Modify [models/inference.py](models/inference.py):

```python
gesture_classifier = GestureInference(..., device="cuda")
```

Or upgrade to a GPU-capable machine.

---

## Workflow Summary

```
1. Record gestures    → python main.py (press 1-5, perform gestures)
                        Saves to: data/gestures/*.npy

2. Train model        → python models/train.py
                        Saves to: models/gesture_classifier_best.pth

3. Evaluate           → python models/evaluate.py
                        Prints: accuracy, precision, recall, F1

4. Perform with       → python main.py
   live prediction       Displays: gesture + confidence in real-time
```

---

## Architecture Summary

### Model

**GestureClassifierLSTM**:

- Input: Sequences of shape `(batch, 60, 18)`
  - 60 frames (∼2 seconds @ 30fps)
  - 18 features (6 joints × 3 coordinates)
- Architecture:
  - 2-layer LSTM (hidden_size=64)
  - Dropout (0.3)
  - Fully connected output → 5 classes
- Parameters: ~15K

### Training

- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)
- **Early stopping**: Patience=10 epochs
- **Data split**: 70% train / 15% val / 15% test
- **Class balancing**: Weighted sampler

### Inference

- Preprocessing: Pad/truncate to 60 frames, flatten to (T, 18)
- Prediction: Forward pass → softmax → argmax
- Output: (label, confidence)

---

## Next Steps

After training a working model, consider:

1. **Bidirectional control**: Use both left + right arms
2. **Advanced features**: Hand position + orientation for UI control
3. **Multi-model ensemble**: Train variants with different architectures
4. **MIDI output**: Connect gestures to orchestration engine
5. **Fine-tuning**: Additional training on user-specific data

---

## Questions?

Refer to inline comments in:

- [models/gesture_dataset.py](models/gesture_dataset.py) — Data loading
- [models/gesture_classifier.py](models/gesture_classifier.py) — Model architecture
- [models/train.py](models/train.py) — Training loop
- [models/inference.py](models/inference.py) — Real-time classification
