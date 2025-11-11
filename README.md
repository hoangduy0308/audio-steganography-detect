# Audio Steganography Project

A comprehensive audio steganography implementation using Least Significant Bit (LSB) technique and Convolutional Neural Networks for detection and analysis.

## Repository

**GitHub**: https://github.com/hoangduy0308/audio_steganography_detect

## Overview

This project implements audio steganography techniques for hiding and detecting hidden messages in audio files. The implementation includes:
- LSB-based audio steganography
- CNN-based steganography detection
- Audio signal processing and analysis
- Training and evaluation frameworks

## Project Structure

```
├── data/                    # Dataset files
├── models/                  # Trained models
├── Baocao_Slide/           # Presentation slides
├── gui_infer.py            # GUI interface for inference
├── tao_dataset.py          # Dataset creation script
├── train_cnn_model.ipynb   # CNN model training notebook
├── train-cnn.ipynb         # Additional CNN training
├── compare_melspec.py      # Mel-spectrogram comparison
├── plot_melspectrogram.py  # Mel-spectrogram visualization
├── plot_waveform.py        # Audio waveform plotting
└── README.md              # This file
```

## Dataset

### Primary Dataset
- **Name**: data_lsb by Hdiiii
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/hdiiii/data-lsb)
- **Size**: 6.02 GB
- **Description**: Dataset containing audio samples with LSB steganography

### Alternative Dataset
- **Name**: UrbanSound8K
- **File**: UrbanSound8K.csv
- **Description**: Urban sound classification dataset

## Requirements

To install the required dependencies, create a `requirements.txt` file with:

```
numpy
pandas
librosa
matplotlib
tensorflow
keras
scikit-learn
torch
torchvision
soundfile
pyaudio
tkinter
jupyter
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/hoangduy0308/audio_steganography_detect.git
cd audio_steganography_detect
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset from Kaggle:
```bash
kaggle datasets download -d hdiiii/data-lsb
```

## Usage

### Training the CNN Model
```bash
jupyter notebook train_cnn_model.ipynb
```

### Running the GUI Application
```bash
python gui_infer.py
```

### Creating Custom Dataset
```bash
python tao_dataset.py
```

### Audio Analysis
```python
# Plot waveform
python plot_waveform.py

# Plot mel-spectrogram
python plot_melspectrogram.py

# Compare mel-spectrograms
python compare_melspec.py
```

## Features

### Steganography Implementation
- **LSB Technique**: Hide messages in the least significant bits of audio samples
- **Multiple Audio Formats**: Support for WAV, MP3, FLAC, and other common formats
- **Customizable Embedding**: Adjustable embedding strength and capacity

### Detection System
- **CNN-Based Detection**: Deep learning model for steganography detection
- **Feature Extraction**: Mel-spectrogram and other audio features
- **Classification**: Binary classification for stego vs. cover audio

### Visualization Tools
- **Waveform Analysis**: Time-domain audio visualization
- **Spectrogram Analysis**: Frequency-domain representation
- **Comparison Tools**: Side-by-side comparison of original and stego audio

## Model Architecture

The CNN model uses:
- **Convolutional Layers**: 2D convolution for spectrogram analysis
- **Pooling Layers**: Max pooling for dimensionality reduction
- **Dropout Layers**: Regularization to prevent overfitting
- **Dense Layers**: Final classification layers

## Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate
- **Recall**: Detection sensitivity
- **F1-Score**: Balance between precision and recall
- **ROC Curve**: Receiver operating characteristic analysis

## File Descriptions

### Core Scripts
- `gui_infer.py`: GUI application for interactive steganography detection
- `tao_dataset.py`: Script to generate training/testing datasets
- `compare_melspec.py`: Compare mel-spectrograms between cover and stego audio
- `plot_melspectrogram.py`: Generate mel-spectrogram visualizations
- `plot_waveform.py`: Create waveform plots for audio analysis

### Training Notebooks
- `train_cnn_model.ipynb`: Main CNN training pipeline
- `train-cnn.ipynb`: Alternative training approach

### Analysis Files
- `UrbanSound8K.csv`: Dataset metadata
- `ketqua.md`: Results summary
- `loi.md`: Error logs
- `tuning_history.md`: Model tuning history

## Results

The project achieves:
- High detection accuracy for LSB steganography
- Robust performance across different audio types
- Efficient processing speed for real-time applications

## Future Work

- [ ] Support for additional steganography techniques
- [ ] Mobile application development
- [ ] Real-time streaming analysis
- [ ] Advanced feature extraction methods
- [ ] Multi-class steganography classification

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle dataset provider Hdiiii for the steganography dataset
- UrbanSound8K dataset for additional audio samples
- Open-source audio processing libraries (librosa, tensorflow, pytorch)

## Contact

For questions or collaboration, please open an issue in this repository.

---

**Note**: This project is for educational and research purposes only. Please ensure compliance with local laws and regulations when using steganography techniques.