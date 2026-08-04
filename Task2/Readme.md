# Task 3: Handwritten Character Recognition
*(Internally organized as Task2 — CodeAlpha ML Internship)*

## Overview
A Convolutional Neural Network (CNN) trained to recognize handwritten characters — digits (0-9) and letters (A-Z, a-z) — using the EMNIST Balanced dataset. Built with TensorFlow/Keras.

## Dataset
- **Source**: EMNIST Balanced (via `tensorflow_datasets`)
- **Classes**: 47 (digits 0-9, uppercase A-Z, and a subset of lowercase letters not visually confusable with their uppercase form)
- **Size**: 112,800 training images, 18,800 test images
- **Balance**: Exactly 2,400 images per class — no class imbalance handling needed (unlike Task 1's credit scoring dataset)
- **Image format**: 28x28 grayscale, originally rotated 90° and horizontally flipped (corrected during preprocessing via `tf.transpose`)

## Approach
1. **EDA**: Verified class balance, visualized sample characters, confirmed correct orientation after transform
2. **Preprocessing**: Normalized pixel values to [0, 1], batched (128), shuffled training data, used `prefetch(AUTOTUNE)` for pipeline efficiency
3. **Model architecture**: 3-block CNN
   - Conv2D(32) → BatchNorm → MaxPool
   - Conv2D(64) → BatchNorm → MaxPool
   - Conv2D(128) → BatchNorm → MaxPool
   - Flatten → Dense(256) → Dropout(0.4) → Dense(47, softmax)
   - ~400K parameters
4. **Training**: 10 epochs, Adam optimizer, sparse categorical crossentropy, trained on CPU (~5 min/epoch)
5. **Evaluation**: Test accuracy, per-class precision/recall/F1, confusion matrix

## Results
- **Test Accuracy**: 88.45%
- **Weighted F1-score**: 0.89

### Per-class performance patterns
Strong performers (F1 > 0.95): distinctive shapes like `3`, `7`, `M`, `W`, `K`, `P`
Weak performers (F1 < 0.70): visually confusable character pairs:
- `0` vs `O` (digit zero vs. capital O)
- `1` vs `I` vs `L` (vertical stroke characters)
- `9` vs `g` vs `q` (loop-with-descender shapes)
- `F` vs `f` (near-identical case forms)

This confirms the model is learning genuine visual features rather than memorizing — its errors mirror the same ambiguities a human reader would face.

## Key Lessons

- **Balanced vs. imbalanced data**: Unlike Task 1 (93/7 skew), EMNIST Balanced has equal class counts, so no `class_weight` or resampling was needed — a good contrast case for understanding when imbalance handling actually matters.
- **Overfitting signal via train/val divergence**: By epoch 7-10, training loss kept dropping while validation loss plateaued/rose slightly — the same overfitting pattern seen with Random Forest depth in Task 1, just showing up as a training-curve divergence instead
- **Save trained models immediately**: A mid-session kernel restart (triggered by installing `seaborn`) wiped all in-memory variables. The trained model itself survived only because the kernel hadn't actually restarted — but this was a close call. Lesson: call `model.save()` immediately after training completes, before running any other cells, so a crash or restart never costs the training time.
- **EMNIST orientation quirk**: Raw EMNIST images are rotated 90° and horizontally flipped compared to human-readable orientation — required `tf.transpose` correction during preprocessing, a good reminder to always visually inspect raw data before trusting it.

## Project Structure
Task2/
├── data/                  # Cached EMNIST datasets (loaded via tensorflow_datasets)
├── models/                # Saved trained model artifacts
│   └── emnist_cnn.keras   # Final trained CNN model in Keras format
├── notebooks/             # Jupyter notebooks for development
│   └── eda.ipynb          # End-to-end pipeline (EDA, preprocessing, CNN, training, eval)
└── README.md              # Project documentation and setup guide
