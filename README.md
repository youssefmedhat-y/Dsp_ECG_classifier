# ECG Heartbeat Classification System

A deep learning system for classifying ECG heartbeats as **Normal** or **PVC (Premature Ventricular Contraction)** using advanced signal processing and the ECGNet neural network architecture.

## 📋 Overview

This project implements a complete ECG classification pipeline that combines:
- **Digital Signal Processing**: Butterworth bandpass filtering and normalization
- **Feature Extraction**: Autocorrelation and Discrete Cosine Transform (DCT)
- **Deep Learning**: ECGNet architecture built with PyTorch

## 🏗️ Architecture

### Signal Processing Pipeline

```
Raw ECG Signal
    ↓
Butterworth Bandpass Filter (0.5-40 Hz)
    ↓
Z-Score Normalization
    ↓
Autocorrelation Feature Extraction
    ↓
Significant Coefficient Selection
    ↓
Discrete Cosine Transform (DCT)
    ↓
Non-Zero Coefficient Selection
    ↓
ECGNet Classification
    ↓
Normal / PVC Prediction
```

### ECGNet Model

The ECGNet is a deep neural network specifically designed for ECG classification:

- **Input Layer**: Feature vector (dimension varies based on signal length)
- **Hidden Layers**:
  - FC1: 256 neurons + BatchNorm + ReLU + Dropout(0.3)
  - FC2: 128 neurons + BatchNorm + ReLU + Dropout(0.3)
  - FC3: 64 neurons + BatchNorm + ReLU + Dropout(0.2)
  - FC4: 32 neurons + BatchNorm + ReLU + Dropout(0.2)
- **Output Layer**: 1 neuron + Sigmoid (binary classification)

**Key Features**:
- Batch normalization for stable training
- Dropout layers for regularization
- Designed for ECG arrhythmia detection tasks

## 📁 Dataset Structure

```
Normal&PVC/
├── Normal_Train.txt    # Training data for normal heartbeats
├── Normal_Test.txt     # Test data for normal heartbeats
├── PVC_Train.txt       # Training data for PVC heartbeats
└── PVC_Test.txt        # Test data for PVC heartbeats
```

**Data Format**: Pipe-delimited text files where each line represents one ECG signal with hundreds of sample points.

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy scipy scikit-learn torch
```

Or simply run the first cell of the notebook - it will auto-install PyTorch if needed.

### Running the Notebook

1. **Open in VS Code**: Open `ecg_classifier.ipynb` in VS Code with Jupyter extension
2. **Or use Google Colab**: Click "Connect to Colab" in the notebook toolbar (Colab extension already installed)
3. **Run All Cells**: Execute cells sequentially from top to bottom

### Quick Start

The notebook is organized into 13 sections:

1. **Import Libraries** - Auto-installs PyTorch if needed
2. **Data Loading** - Functions to load pipe-delimited ECG data
3. **Preprocessing** - Butterworth filter and normalization
4. **Feature Extraction** - Autocorrelation and DCT
5. **ECGNet Architecture** - Define the neural network model
6. **Load Training Data** - Load Normal and PVC training samples
7. **Extract Training Features** - Process training data through pipeline
8. **Train Model** - Train ECGNet with validation split
9. **Load Test Data** - Load test samples
10. **Extract Test Features** - Process test data
11. **Evaluate Model** - Confusion matrix and metrics
12. **Save Model** - Save trained PyTorch checkpoint
13. **Predict Samples** - Test on individual ECG signals

## 📊 Expected Results

The model achieves high accuracy in distinguishing between Normal and PVC heartbeats:

- **Accuracy**: >95% (typical)
- **Precision**: High precision for both classes
- **Recall**: Balanced recall across Normal and PVC detection

**Confusion Matrix Output**:
```
                Predicted
              Normal  PVC
Actual Normal   XXX    XX
       PVC       XX   XXX
```

## 🔧 Key Functions

### Preprocessing
- `load_ecg_data(filepath)` - Load pipe-delimited ECG data
- `butterworth_bandpass_filter(data, fs=360, lowcut=0.5, highcut=40)` - Apply bandpass filter
- `normalize_signals(data)` - Z-score normalization

### Feature Extraction
- `extract_autocorrelation(data, max_lag)` - Compute autocorrelation features
- `select_significant_coefficients(autocorr, threshold=0.1)` - Filter significant coefficients
- `extract_dct_features(data)` - Apply Discrete Cosine Transform
- `select_nonzero_coefficients(dct_features, threshold=1e-6)` - Select non-zero DCT coefficients
- `preprocess_and_extract_features(data)` - Complete pipeline

### Model & Prediction
- `ECGNet(input_dim)` - PyTorch model class
- `predict_single_ecg(ecg_signal, model, device)` - Predict single ECG sample

## 💾 Model Saving & Loading

### Save Model
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': X_train.shape[1],
    'optimizer_state_dict': optimizer.state_dict(),
}, 'ecgnet_classifier.pth')
```

### Load Model
```python
checkpoint = torch.load('ecgnet_classifier.pth')
model = ECGNet(checkpoint['input_dim']).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 🧪 Signal Processing Details

### Butterworth Bandpass Filter
- **Frequency Range**: 0.5 Hz (low cutoff) to 40 Hz (high cutoff)
- **Order**: 4th order
- **Purpose**: Remove baseline wander and high-frequency noise from ECG signals
- **Sampling Frequency**: 360 Hz (standard for ECG databases)

### Autocorrelation Analysis
- **Purpose**: Capture temporal patterns and periodicities in ECG signals
- **Method**: Normalized autocorrelation with positive lags
- **Significance Threshold**: 0.1 (coefficients below this are zeroed)

### Discrete Cosine Transform (DCT)
- **Purpose**: Transform time-domain features to frequency domain
- **Normalization**: Orthonormal DCT
- **Selection**: Only non-zero coefficients retained (threshold: 1e-6)

## 🎯 Use Cases

- **Arrhythmia Detection**: Identify abnormal heart rhythms
- **Clinical Decision Support**: Assist cardiologists in diagnosis
- **Real-time Monitoring**: ECG monitoring systems
- **Research**: Study PVC patterns and characteristics

## 📝 Training Parameters

- **Epochs**: 50
- **Batch Size**: 32
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Binary Cross Entropy (BCE)
- **Train/Validation Split**: 80/20
- **Device**: Automatic GPU detection (CUDA if available, else CPU)

## 🔍 Model Evaluation Metrics

The notebook provides comprehensive evaluation:

1. **Confusion Matrix** - Visual breakdown of predictions
2. **Classification Report** - Precision, recall, F1-score per class
3. **Accuracy** - Overall classification accuracy
4. **Precision** - True positive rate for PVC detection
5. **Recall** - Sensitivity in detecting PVC cases

## 🤝 Contributing

To improve this project:
1. Experiment with different filter parameters
2. Try alternative feature extraction methods
3. Tune ECGNet hyperparameters
4. Add data augmentation techniques
5. Implement cross-validation

## 📚 References

- **ECG Signal Processing**: Butterworth filtering for biomedical signals
- **Feature Extraction**: Autocorrelation and DCT for time-series analysis
- **ECGNet Architecture**: Deep learning for ECG arrhythmia classification

## 🔗 Related Technologies

- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **SciPy**: Signal processing (Butterworth filter, DCT)
- **Scikit-learn**: Metrics and evaluation

## 📄 License

This project is for educational and research purposes.

## 🙋 Support

For issues or questions:
1. Check that data files are in the correct `Normal&PVC/` directory
2. Ensure all dependencies are installed
3. Verify PyTorch installation and CUDA availability (if using GPU)

---

**Built with ❤️ for ECG classification and arrhythmia detection**
