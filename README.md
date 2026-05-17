# Musical Instrument Classification with CNNs

A deep learning audio classification project focused on recognizing musical instruments from raw audio recordings using MFCC representations and convolutional neural networks.

The project implements the complete pipeline:

* audio preprocessing,
* envelope-based signal cleaning,
* FFT and filter bank analysis,
* MFCC extraction,
* CNN training with Keras,
* window-based inference and prediction export.

The project combines classical digital signal processing techniques with deep learning for robust musical instrument recognition.

## Classification pipeline

### Overview

The goal of this project is to automatically classify musical instruments from audio recordings.

Since audio is a continuous temporal signal, the pipeline first transforms raw waveforms into structured frequency-based representations before training a convolutional neural network capable of learning discriminative acoustic patterns.

Instead of using raw audio directly, the model operates on MFCCs (Mel-Frequency Cepstral Coefficients), which provide a compact and efficient representation of timbre-related information.

---

## Dataset

The dataset comes from a Kaggle musical instrument classification competition.

Each audio file is associated with a label corresponding to an instrument category such as:

* piano,
* guitar,
* violin,
* drums,
* flute,
* saxophone,
* and others.

---

## Preprocessing and Feature Engineering

Before training, several signal processing techniques are applied to better understand and prepare the data.

### Envelope-based cleaning

An envelope detection algorithm is used to automatically remove silent regions and retain only meaningful audio content.

This preprocessing step improves training quality by reducing irrelevant information.

### Spectral analysis

The project includes multiple audio representations:

* temporal waveform,
* Fourier Transform (FFT),
* filter banks,
* MFCCs.

MFCCs are ultimately used as model inputs because they summarize perceptually relevant frequency information while producing a 2D representation that is well suited for convolutional neural networks.

---

## Model

The classifier is built using TensorFlow/Keras and consists of a convolutional neural network composed of:

* convolutional layers,
* max pooling layers,
* dropout regularization,
* dense classification layers.

The CNN learns local patterns inside the MFCC representations similarly to how image CNNs learn visual features.

---

## Training

Training is performed using randomly sampled audio windows extracted from the recordings.

This strategy increases sample diversity and improves generalization.

The training pipeline also includes:

* normalization,
* class weighting,
* shuffled mini-batches,
* categorical classification.

The model reaches approximately **93% accuracy** on the validation set.

---

## Prediction

During inference:

1. Audio files are split into successive temporal windows.
2. Each window is converted into MFCCs.
3. Features are normalized using the same preprocessing pipeline as training.
4. Predictions are generated for every window.
5. Probabilities are averaged across the full audio file.

Final predictions are exported into a CSV file.

---

## Project Structure

```text
.
├── requirements.txt
├── cfg.py
├── eda.py
├── model.py
├── predict.py
├── instruments.csv
├── predictions.csv
└── /wavfiles
```

---

# Running the Project

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

---

## 2. Install dependencies

It is recommended to create a virtual environment first.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

---

## 3. Prepare the dataset


The metadata CSV file should match the paths expected in `instruments.csv` for the folder `wavfiles/`.

---

## 4. Run exploratory analysis and preprocessing

```bash
python eda.py
```

This step generates cleaned audio samples and preprocessing artifacts.

---

## 5. Train the CNN model

```bash
python model.py
```

The trained model will be exported into the `models/` directory.

---

## 6. Run predictions

```bash
python predict.py
```

Prediction results will be saved into:

```text
predictions.csv
```

---

# Notes

* The project mixes classical DSP techniques with deep learning.
* MFCCs are treated as image-like representations for CNN processing.
* Window-based prediction averaging improves inference stability.
* Envelope cleaning significantly reduces silence-related noise.
* The project was designed as an end-to-end introduction to audio classification pipelines.


