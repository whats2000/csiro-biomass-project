# CSIRO Pasture Biomass Prediction

GPU-accelerated PyTorch pipeline for predicting pasture biomass from field images and tabular features.

## Project Goal

Build a production-ready model that predicts pasture biomass (grams) for each `(image, target_name)` pair using:
- Field images
- Tabular covariates (NDVI, height, state, species, sampling date)
- Pre-trained vision backbone + tabular fusion head

The model optimizes for the competition's **global weighted R²** metric.

## Setup

This project uses `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

## Project Structure

```
├── data/               # Data files (train.csv, test.csv, images)
├── src/                # Source code
│   ├── data/          # Data loading and preprocessing
│   ├── models/        # Model architectures
│   ├── training/      # Training and evaluation
│   └── utils/         # Utility functions
├── notebooks/         # Jupyter notebooks for exploration
├── scripts/           # Training and inference scripts
└── outputs/           # Model checkpoints and predictions
```

## Usage

See `AGENTS.md` for detailed implementation instructions.

## Target Variables

The model predicts 5 biomass targets (in grams):
- `Dry_Green_g` (weight: 0.1)
- `Dry_Dead_g` (weight: 0.1)
- `Dry_Clover_g` (weight: 0.1)
- `GDM_g` (Green Dry Matter, weight: 0.2)
- `Dry_Total_g` (weight: 0.5)

## License

See LICENSE file for details.
