# EVA — Emergency Voice Activated Bracelet

**EVA (Emergency Voice Activated Bracelet)** is a university project developed for the Digital Systems Design course at Politecnico di Milano. It explores a voice-activated bracelet to support requests for help in dangerous situations, with particular attention to violence against women. The project investigates an **embedded keyword spotting** system that detects the Italian activation phrase **“EVA aiutami”** (“EVA, help me”), allowing the user to initiate an emergency request without manual interaction.

The system is designed to acquire audio from the digital microphone of the **XIAO ESP32-S3 Sense** and run inference locally on its microcontroller. Model selection considers recognition performance alongside network size and computational requirements, with the aim of identifying a classifier suitable for the memory and processing constraints of the ESP32-S3.

This repository contains the **Python/TensorFlow experimentation pipeline** used to develop and compare candidate models for embedded deployment. Audio windows are converted into **Mel-Frequency Cepstral Coefficients (MFCCs)** and classified using **1D or 2D convolutional neural networks (CNNs)**. The code supports individual experiments and grid search with grouped cross-validation, allowing acoustic feature parameters and network architectures to be compared under a common evaluation protocol. Training runs on the computer; inference latency, memory usage, and energy consumption must be measured separately on the target hardware.

## Dataset and acquisition

The dataset was collected from **multiple participants**, across recording sessions and acoustic conditions. All recordings included in the dataset were acquired using the **digital microphone of the XIAO ESP32-S3 Sense**, matching the audio acquisition hardware intended for the prototype.

Recordings are organized into three classes:

| Class | Content |
|---|---|
| `eva` | The complete activation phrase **“EVA aiutami”** |
| `unknown` | Other words, phrases, and conversations that do not constitute the activation phrase |
| `noise` | Environmental sounds without speech, according to the adopted annotation criteria |

The collection includes different voices and noise environments to assess performance on speakers and environments excluded from training.

Initial board recordings were saved as 10-second WAV files. Following annotation, recordings were segmented into **1100 ms samples**, checking that positive samples contained the complete activation phrase. The pipeline uses **mono WAV files with 16-bit PCM encoding and a 16 kHz sampling rate**.

Filenames preserve **speaker (`Pxx`)**, **environment (`Axx`)**, session, and recording identifiers to retain sample traceability. Class labels come from annotation; speaker and environment identifiers are used for grouping, not for inferring a label.

The code supports two classification tasks:

- **Three-class classification:** `eva`, `unknown`, and `noise`.
- **Binary classification:** `eva` versus `non_target`, formed by merging `unknown` and `noise` in memory. Switching modes does not require renaming the WAV files.

Audio recordings and experiment outputs are not distributed with this repository. They remain in the local working directory and are excluded from version control through `.gitignore`.

## Sample preparation with Edge Impulse

**[Edge Impulse Studio](https://docs.edgeimpulse.com/tutorials/end-to-end/keyword-spotting)** is used to support dataset preparation: importing recordings, assigning labels, and segmenting recordings into 1100 ms samples. The prepared WAV files are exported for use in the local pipeline. Training, grid search, and evaluation described in this repository run in Python/TensorFlow.

Separately from the use of Studio for dataset preparation, the MFCC implementation incorporates DSP components derived from Edge Impulse sources, preserved in `third_party/` with their license notices. Numerical equivalence with a particular Edge Impulse project must be checked separately using matching DSP settings.

## How the pipeline works

1. **Audio indexing and validation.** The code indexes WAV files, checks format and duration, and extracts the identifiers needed for group separation. Integrity checks include detecting duplicate audio.
2. **Grouped splitting.** Speakers and environments are assigned to folds while keeping all recordings from the same group together, regardless of session or device, to prevent data leakage.
3. **MFCC extraction.** Each sample is transformed into a sequence of coefficient vectors used as network input. Features are cached to avoid repeated computation. MFCCs are also calculated for validation and test samples: this DSP transformation is not learned from training labels.
4. **CNN training.** The network learns to distinguish the classes. Both 1D and 2D CNNs are available, with configurable filters, kernels, depth, and dropout. Augmentation applies Gaussian noise and temporal masking only during training.
5. **Checkpoint selection.** Validation loss determines which model weights are retained for evaluation. With the current settings, training runs all configured epochs, including those after the best checkpoint, so that learning curves share the same epoch range.
6. **Evaluation and export of results.** The selected checkpoint is evaluated on the fold's test set. The code saves metrics, predictions, configurations, and learning histories; grid mode does not generate plots.

## Code organization

| File or directory | Purpose |
|---|---|
| `train.py` | Entry point: user settings, single experiments, and custom grid search |
| `config.py` | Data, MFCC, network, and training configuration definitions |
| `data.py` | Dataset indexing, label handling, and integrity checks |
| `features.py` | MFCC extraction and feature caching |
| `models.py` | 1D and 2D CNN construction |
| `experiments.py` | Individual model training, checkpoints, and experiment saving |
| `cross_validation.py` | Speaker/environment fold construction and execution |
| `metrics.py`, `keras_metrics.py` | Evaluation metrics and EVA AUC during training |
| `cv_reports.py` | Metric aggregation and plot generation |
| `third_party/` | External DSP components and their licenses |

## Installation and preparation

The pipeline requires a Python environment with TensorFlow, NumPy, SciPy, scikit-learn, and Matplotlib. Development uses **Spyder with Anaconda**; scripts can also run from a terminal. Dependencies are listed in `requirements.txt`.

In a dedicated environment:

```bash
python -m pip install -r requirements.txt
```

When using an existing working Anaconda environment, check installed dependencies before reinstalling packages.

Organize local WAV files as follows:

```text
data_v2/
├── training/
│   ├── eva/
│   ├── unknown/
│   └── noise/
└── test/
    ├── eva/
    ├── unknown/
    └── noise/
```

Prepare the labeled 1100 ms WAV samples before running this training pipeline, retaining speaker and environment identifiers in the filenames. Dataset preparation and diagnostic utilities are maintained locally and are not included in this repository.

Set `data_dir` in `train.py` to the dataset directory. The training pipeline checks audio format, duration, grouping, and duplicates as part of loading and evaluation.

When cross-validation is enabled, the historical `training` and `test` directories are pooled in memory and split again by group. WAV files are not moved.

## Running a single experiment

In `train.py`, set:

```python
MODE = 'single'
```

Edit `SETTINGS`, `SINGLE_FEATURE`, and `SINGLE_MODEL`. For example:

```python
SINGLE_FEATURE = MFCCConfig(n_mfcc=13, n_mels=32)
SINGLE_MODEL = ModelConfig(
    family='cnn1d',
    filters=(8, 16),
    kernel=5,
    dropout=0.25,
)
```

`SETTINGS` controls binary or three-class classification (`binary`), learning rate, batch size, epochs, seeds, and augmentation. Each element in `filters` defines a convolutional block: `(8, 16)` creates two blocks, while `(8, 16, 32)` creates three. For 2D CNNs, the kernel is square: `kernel=5` means 5 × 5.

To disable augmentation, set **both** `gaussian_std=0.0` and `time_mask_max_frames=0`. Dropout is controlled separately in the network configuration.

In Spyder, run the complete `train.py` file with **F5**. From a terminal:

```bash
python train.py
```

Single mode evaluates one feature/network configuration across the configured folds and generates per-fold and aggregate plots: loss versus epochs, EVA ROC AUC versus epochs, and confusion matrices.

## Comparing configurations with grid search

In `train.py`, set:

```python
MODE = 'custom_grid'
```

The grid evaluates the product of the configured lists:

| List | Parameters compared |
|---|---|
| `FEATURES` | MFCC coefficient count, Mel filter count, and other DSP parameters |
| `GRID_FAMILIES` | 1D and 2D CNNs |
| `GRID_KERNELS` | Convolutional kernel sizes |
| `GRID_FILTERS` | Filter counts and number of convolutional blocks |
| `GRID_LEARNING_RATES` | Learning rates |

`MODELS` is constructed by combining network family, filter configuration, and kernel size. Grid dropout is set to 0.25; batch size, augmentation, and other settings come from `SETTINGS`. Grid learning rates come from `GRID_LEARNING_RATES`, rather than the learning rate used in single mode.

Shorten the lists to reduce the number of experiments. The total training count is printed before execution:

```text
MFCC configurations × networks × learning rates × folds × seeds
```

Grid mode does not open plot windows or generate PNGs for each experiment. It saves metrics and numerical learning histories for subsequent comparison. Completed experiments with identical configurations and data are reused; interrupted experiments restart from the beginning.

## Cross-validation and interpretation

The current configuration uses **five grouped folds**: each speaker or environment appears in the test set once per repetition. Within each fold, some of the remaining groups are reserved for validation and checkpoint selection. Split proportions are approximate because groups may differ in size.

Outputs include EVA precision, recall, F1, and AUC, together with accuracy, loss, and confusion matrices. Aggregate performance is reported as the **mean and standard deviation across folds**. Multiple seeds can be configured to assess training variability as well.

Files are saved under `results/`. Grid results are separated by learning rate, with a combined summary at:

```text
results/custom_grid/grid_summary_all_learning_rates.csv
```

Mean learning curves show progression across epochs. A horizontal test line, when present, represents the mean performance of the selected checkpoints, not repeated test evaluation at every epoch.

Comparisons should consider both missed detections and false positives. Errors on 1100 ms clips do not directly measure false alarms during continuous listening. Repeatedly using outer-fold results to select models also makes those results part of development rather than independent final validation.

## Third-party DSP components

The `third_party/` directory is a required runtime dependency: `features.py` imports its feature extraction and signal processing modules to compute MFCCs. It must be distributed with the training code, including its source attribution and license files. It does not contain private recordings or trained models.

Development tests and dataset preparation utilities are kept locally and excluded from the published training repository.
