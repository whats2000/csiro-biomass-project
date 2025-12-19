# CSIRO Pasture Biomass Prediction

GPU-accelerated PyTorch pipeline for predicting pasture biomass from field images and tabular features.

## Project Goal

This project implements a solution for the [CSIRO Pasture Biomass Prediction](https://www.kaggle.com/competitions/csiro-biomass/) Kaggle competition.

Build a production-ready model that predicts pasture biomass (grams) for each `(image, target_name)` pair using:
- Field images
- Tabular covariates (NDVI, height, state, species, sampling date)
- Pre-trained vision backbone + tabular fusion head

The model optimizes for the competition's **global weighted R²** metric.

## Features

- **Multimodal Fusion**: Combines vision features from images with tabular data
- **Multiple Architectures**: Simple concatenation, cross-attention, and vision-only models
- **Cross-Validation**: 5-fold CV for robust evaluation
- **GPU Acceleration**: Mixed precision training with automatic device selection
- **Production Ready**: Modular design with configuration files and logging

## Installation

This project uses `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### Verify Setup

Run the setup verification script:

```bash
python scripts/verify_setup.py
```

## Data Preparation

Place your data files in the `data/` directory:
- `train.csv`: Training data with image paths and targets
- `test.csv`: Test data for inference
- `sample_submission.csv`: Submission format template
- `train/` and `test/`: Directories containing the images

The data should include:
- **Images**: Field photographs of pasture
- **Tabular Features**:
  - Categorical: `State`, `Species`
  - Continuous: `Pre_GSHH_NDVI`, `Height_Ave_cm`
- **Targets**: 5 biomass measurements (grams)

## Model Architectures

### 1. Simple Concatenation (Baseline)
- Concatenates vision and tabular embeddings
- Lightweight and fast training
- Good starting point for experimentation

### 2. Multimodal Attention Fusion
- Cross-attention between vision and tabular features
- Transformer blocks for enhanced fusion
- Best performance for complex relationships

### 3. DINO Vision-Only
- Uses DINOv2 backbone for vision features only
- No tabular fusion
- Specialized for vision-dominant tasks

## Training

Train models using the provided configuration files:

```bash
# Train simple concatenation model
python scripts/train.py --config configs/train_simple_concat.yaml

# Train multimodal attention model
python scripts/train.py --config configs/train_multimodal_attention.yaml

# Train DINO vision-only model
python scripts/train.py --config configs/train_dino.yaml
```

### Custom Training

You can also modify configurations or create new ones. Key parameters:
- `model.name`: Choose from `simple_concat`, `multimodal_attention`, `dino_vision`
- `model.backbone`: Vision backbone (e.g., `efficientnet_b0`, `vit_base_patch14_dinov2`)
- `training.n_folds`: Number of CV folds
- `training.epochs`: Training epochs per fold

## Inference

Generate predictions for test data:

```bash
# Run inference with trained model
python scripts/inference.py --config configs/inference_multimodal_attention.yaml
```

Inference configs correspond to training configs and will automatically find the best checkpoint.

## Evaluation

The project uses **weighted R²** as the primary metric:

- **Targets**: `Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`, `GDM_g`, `Dry_Total_g`
- **Weights**: 0.1, 0.1, 0.1, 0.2, 0.5 respectively
- **Derivations**:
  - `GDM_g = Dry_Green_g + Dry_Clover_g`
  - `Dry_Total_g = Dry_Green_g + Dry_Dead_g + Dry_Clover_g`

## Exploratory Data Analysis

Run EDA to understand the dataset:

```bash
python scripts/eda.py
```

This will generate statistics, correlations, and visualizations.

## Project Structure

```
├── configs/               # YAML configuration files
├── data/                  # Dataset files and images
├── notebooks/             # Jupyter notebooks (empty)
├── outputs/               # Model checkpoints and predictions
├── scripts/               # Training, inference, and utility scripts
│   ├── train.py           # Main training script
│   ├── inference.py       # Inference script
│   ├── eda.py             # Exploratory data analysis
│   └── verify_setup.py    # Setup verification
└── src/biomass/           # Source code package
    ├── data/              # Data loading and preprocessing
    ├── models/            # Model architectures
    ├── training/          # Training utilities and metrics
    └── utils/             # Configuration and utilities
```

## Configuration

Configuration files control all aspects of training and inference. See `configs/` for examples.

Key sections:
- `data`: Data paths and settings
- `model`: Architecture and hyperparameters
- `training`: Training parameters
- `image`: Image preprocessing settings

## Requirements

- Python 3.10-3.12
- CUDA-compatible GPU (recommended)
- 16GB+ RAM for training
- Dependencies managed via `uv` (see `pyproject.toml`)

## License

See LICENSE file for details.
