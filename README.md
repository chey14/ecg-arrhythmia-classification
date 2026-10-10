# ECG Arrhythmia Classification Using Machine Learning and Deep Learning

A machine learning project that classifies individual ECG heartbeats into five AAMI-inspired beat categories using a Random Forest and a 1D Convolutional Neural Network (CNN). The project includes ECG preprocessing, model comparison, recording-wise grouped cross-validation, and an interactive Streamlit demo.

## Overview

Cardiac arrhythmias are irregularities in the heart's electrical activity. Electrocardiograms (ECGs) provide a signal that can be analyzed to identify different heartbeat patterns.

In this project, I built an end-to-end experimental pipeline using the **MIT-BIH Arrhythmia Database**. I extracted annotated heartbeat segments, trained multiple machine learning models, evaluated their performance using held-out recordings, and developed a Streamlit interface to visualize and classify individual heartbeats.

The project focuses on understanding model performance beyond accuracy, particularly when heartbeat classes are highly imbalanced.

> **Disclaimer:** This is an educational machine learning prototype, not a medical device. Its predictions must not be used for medical diagnosis or treatment.

## Features

- ECG heartbeat extraction using WFDB annotations
- Mapping of annotation symbols into five AAMI-inspired categories
- Fixed-length heartbeat segments containing 252 samples
- Statistical feature extraction for a feature-based baseline
- Random Forest classification using raw ECG segments
- A 1D CNN trained with focal loss to address class imbalance
- Recording-wise grouped cross-validation
- Evaluation using accuracy, macro F1, per-class precision and recall, and confusion matrices
- Interactive Streamlit application for waveform visualization and heartbeat classification
- Saved out-of-fold predictions for model comparison and error analysis

## Tech stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy and Pandas | Data manipulation and analysis |
| WFDB | Reading ECG recordings and annotations |
| SciPy | Scientific signal-processing utilities |
| scikit-learn | Random Forests, cross-validation, and evaluation metrics |
| PyTorch | 1D CNN implementation and training |
| Matplotlib | ECG waveform visualization |
| Streamlit | Interactive web application |
| Git and GitHub | Version control and project hosting |

## Dataset

This project uses the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/), available through PhysioNet.

The database contains ECG recordings with expert beat annotations. The current implementation uses **15 recordings and 33,889 annotated heartbeat segments**.

Each extracted heartbeat is represented by 252 samples:
- 90 samples before the annotated beat location
- 162 samples from the annotated location onward

At the database's 360 Hz sampling rate, each segment covers approximately 0.7 seconds.

### Beat categories

The annotation symbols are mapped into five categories inspired by the AAMI beat-class grouping:

| Label | Description | Example source symbols |
|---|---|---|
| N | Normal and related beats | `N`, `L`, `R`, `e`, `j` |
| S | Supraventricular ectopic beats | `A`, `a`, `J`, `S` |
| V | Ventricular ectopic beats | `V`, `E` |
| F | Fusion beats | `F` |
| Q | Unknown or paced-related category | `/`, `f`, `Q` |

The original annotation symbols are grouped into these categories before training. The mapping is based on the project's implemented label dictionary and should not be interpreted as a clinical diagnosis.

### Class distribution

The extracted dataset is highly imbalanced:

| Class | Number of beats | Approximate share |
|---|---:|---:|
| N | 30,238 | 89.2% |
| S | 159 | 0.47% |
| V | 3,094 | 9.1% |
| F | 389 | 1.15% |
| Q | 9 | 0.03% |
| **Total** | **33,889** | **100%** |

This imbalance is a central challenge in the project. A high overall accuracy can hide poor recognition of rare heartbeat classes.

## Project workflow

The pipeline follows these stages:

1. **Load ECG data:** Read recordings and beat annotations using WFDB.
2. **Extract heartbeat segments:** Use annotated beat locations to select fixed-length signal windows.
3. **Prepare labels:** Map the original annotation symbols into five categories.
4. **Build input representations:** Use raw waveform samples for the CNN and raw-signal Random Forest, and nine statistical features for the feature-based Random Forest.
5. **Train models:** Train and compare multiple classification approaches.
6. **Evaluate models:** Use recording-wise grouped cross-validation so a recording is not split between training and testing within a fold.
7. **Analyze errors:** Compare per-class precision, recall, macro F1, and confusion matrices.
8. **Visualize predictions:** Use Streamlit to display a heartbeat waveform, its predicted category, and its reference annotation.

**Scope limitation:** The current pipeline classifies beats located using existing dataset annotations. It does not automatically detect heartbeats in a continuous, previously unseen ECG recording.

## Models

### 1. Raw-signal Random Forest

A Random Forest classifier trained directly on the fixed-length ECG waveform samples.

This provides a baseline for assessing whether an ensemble of decision trees can learn useful patterns from the signal.

### 2. Statistical-feature Random Forest

A Random Forest trained using nine statistical features extracted from each heartbeat:

- Mean
- Standard deviation
- Minimum
- Maximum
- Peak-to-peak amplitude
- Median
- 25th percentile
- 75th percentile
- Root mean square (RMS)

This experiment investigates whether simple statistical summaries retain enough information for heartbeat classification.

### 3. One-dimensional CNN with focal loss

A lightweight 1D CNN learns local patterns directly from the ECG waveform.

The network uses:
- One-dimensional convolutional layers
- Batch normalization and ReLU activations
- Max pooling
- Adaptive average pooling
- Fully connected classification layers
- Dropout for regularization

The CNN uses per-heartbeat normalization and focal loss with class weights to place greater emphasis on difficult examples and address the imbalance between frequent and rare classes.

The model was trained using PyTorch. On Apple Silicon, the implementation can use the Metal Performance Shaders (MPS) backend when available, with CPU as a fallback.

## Evaluation methodology

### Recording-wise grouped cross-validation

The current evaluation uses five-fold `GroupKFold`, with recording identifiers used as groups. Each recording is held out as a test group in one fold and excluded from that fold's training set.

Out-of-fold predictions are collected for aggregate evaluation.

This is more informative than randomly splitting individual heartbeats, because adjacent beats from the same recording can share recording-specific characteristics.

**Important limitation:** The current experiment has not verified that the held-out recordings correspond to entirely unseen patients. The results should therefore be described as *recording-wise grouped cross-validation*, not patient-independent validation or the standard DS1/DS2 inter-patient evaluation.

### Metrics

The project reports:

- **Accuracy:** Fraction of all beats classified correctly.
- **Precision:** Fraction of predictions for a class that are correct.
- **Recall:** Fraction of true examples of a class that are identified.
- **Macro F1:** Unweighted average of F1 scores across the five classes.
- **Confusion matrix:** Counts of true and predicted classes.

Macro F1 is particularly useful here because it gives each class equal weight, rather than allowing the dominant normal class to determine the overall score.

## Results

The following results were obtained from the existing grouped cross-validation prediction files.

| Model | Accuracy | Macro F1 | Macro precision | Macro recall |
|---|---:|---:|---:|---:|
| CNN with focal loss | 86.46% | **0.3006** | 0.2820 | 0.3475 |
| Raw-signal Random Forest | 86.95% | 0.2907 | 0.2761 | 0.3179 |
| Statistical-feature Random Forest | 87.05% | 0.2649 | 0.2638 | 0.2670 |

### Interpretation

The CNN with focal loss achieved the highest macro F1 among the three evaluated models. The raw-signal Random Forest achieved slightly higher accuracy, while the statistical-feature Random Forest achieved the highest accuracy overall but had the lowest macro F1.

The CNN's recall for the ventricular class (V) was approximately 85%, compared with approximately 68% for the raw-signal Random Forest and 39% for the statistical-feature Random Forest.

However, **all three models failed to correctly identify the S class in the current out-of-fold predictions**, and performance on the F and Q classes was also poor.

These results show why accuracy alone is insufficient for imbalanced classification. They also indicate that the current models are experimental baselines rather than reliable classifiers for all five categories.

The CNN's saved full-data model is intended for the interactive demonstration. Its training loss is not an independent estimate of generalization performance; the cross-validation results above are the evaluation results to report.

## Streamlit demo

The application provides an interface for exploring the extracted dataset:

- Select a recording.
- Select an annotated heartbeat segment.
- View its ECG waveform.
- Run the trained CNN.
- View the predicted category and reference annotation.
- Inspect the model's output scores.

The scores produced by the CNN's softmax layer are not clinically calibrated probabilities.

The application currently operates on heartbeat segments from the prepared dataset. Uploading arbitrary continuous ECG recordings and detecting their beats automatically are not yet implemented.

## Repository structure

```text
ecg-arrhythmia-classification/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── mit-bih/
│   └── processed/
├── models/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── focal_loss.py
│   ├── download_data.py
│   ├── build_dataset.py
│   ├── prepare_split.py
│   ├── train_random_forest.py
│   ├── train_feature_rf_cv.py
│   ├── train_cnn_cv.py
│   ├── train_cnn_final.py
│   ├── evaluate.py
│   ├── evaluate_grouped_cv.py
│   └── compare_models.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `data/` and `models/` directories contain local datasets, generated predictions, and trained weights. Their contents are excluded from Git by the repository's ignore rules. The exact files available locally may differ from this illustrative structure.

## Installation and setup

### Prerequisites

- Python 3.10 or newer
- Git
- Internet access to obtain the dataset
- A few gigabytes of free disk space are recommended

A GPU is not required. Apple Silicon users can use MPS if supported by their PyTorch installation.

### 1. Clone the repository

```bash
git clone https://github.com/chey14/ecg-arrhythmia-classification.git
cd ecg-arrhythmia-classification
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the ECG records

```bash
python -m src.download_data
```

The download script fetches the records configured in the project.

### 5. Build the heartbeat dataset

```bash
python -m src.build_dataset
```

This creates the processed heartbeat dataset used by the training scripts.

### 6. Run the model comparison experiments

For the raw-signal Random Forest:

```bash
python -m src.evaluate_grouped_cv
```

For the statistical-feature Random Forest:

```bash
python -m src.train_feature_rf_cv
```

For the CNN with focal loss:

```bash
python -m src.train_cnn_cv
```

To compare the saved out-of-fold predictions:

```bash
python -m src.compare_models
```

The comparison script writes its summary to `data/processed/model_comparison.csv`.

### 7. Train the CNN for the demo

```bash
python -m src.train_cnn_final
```

This saves the full-data demo model to `models/ecg_cnn_final.pth`.

The full-data model is intended for inference in the demo; it should not be evaluated on the same training examples and presented as an independent test result.

### 8. Launch Streamlit

```bash
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit in your browser.

**Note:** The dataset and trained model weights are not included in the repository. Run the preparation and training steps before launching the app.

## Reproducibility and data management

- The experiments use fixed random seeds where configured.
- Grouped cross-validation holds out complete recordings rather than randomly splitting beats.
- The saved prediction files support repeatable metric calculation and error analysis.
- The dataset, processed arrays, and trained model weights are excluded from Git to keep the repository lightweight.
- Results can vary if the dataset, split strategy, preprocessing, or model configuration changes.

The current evaluation does not establish patient-independent generalization. A future improvement is to implement and verify the standard inter-patient DS1/DS2 split before making patient-independent performance claims.

## Challenges and lessons learned

### Class imbalance

Normal beats dominate the dataset. Experiments with class weighting and focal loss helped explore how the training objective affects minority-class recognition, but they did not solve the problem completely.

### Accuracy versus macro F1

The statistical-feature Random Forest achieved the highest accuracy, but its macro F1 was lower than the CNN's. This demonstrated the importance of evaluating each class rather than relying on a single aggregate score.

### Model comparison

The raw-signal Random Forest, statistical-feature Random Forest, and CNN provided different trade-offs in accuracy, macro F1, and ventricular-class recall. The results support comparing models experimentally rather than assuming that a deep learning model will always perform best.

### Evaluation design

Grouping by recording prevents the same recording from appearing in both training and test sets within a fold. Verifying patient identities and using a standard inter-patient split are necessary next steps for a stronger claim about generalization to unseen patients.

## Future improvements

- Implement and verify the standard DS1/DS2 inter-patient evaluation protocol.
- Add a band-pass filtering stage and ensure consistent preprocessing throughout the pipeline.
- Build an RR-interval baseline using beat timing features.
- Explore a CNN that combines waveform features with RR intervals.
- Improve minority-class performance through carefully controlled training experiments.
- Add per-class error analysis and ECG strips colored by predicted class.
- Support user-uploaded ECG recordings after implementing and validating beat detection and sampling-rate handling.
- Add automated tests and document reproducible experiment configurations.

## What I learned

This project provided hands-on experience with:

- Processing biomedical time-series data
- Reading ECG signals and annotations
- Building machine learning and deep learning pipelines
- Implementing a 1D CNN in PyTorch
- Working with imbalanced multi-class datasets
- Using grouped cross-validation
- Comparing models with macro F1 and per-class metrics
- Building a Streamlit application for model inference
- Managing experiments and code with Git and GitHub

## References

1. Moody, G. B., and Mark, R. G. *The impact of the MIT-BIH Arrhythmia Database*. IEEE Engineering in Medicine and Biology Magazine.
2. [MIT-BIH Arrhythmia Database — PhysioNet](https://physionet.org/content/mitdb/)
3. [WFDB Python package](https://wfdb.readthedocs.io/)
4. [PyTorch documentation](https://pytorch.org/docs/)
5. [scikit-learn documentation](https://scikit-learn.org/stable/)
6. de Chazal, P., O'Dwyer, M., and Reilly, R. B. *Automatic Classification of Heartbeats Using ECG Morphology and Heartbeat Interval Features*. IEEE Transactions on Biomedical Engineering, 2004.

