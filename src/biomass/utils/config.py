"""Configuration management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    """Configuration for biomass prediction pipeline."""
    
    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    train_csv: Optional[Path] = None
    test_csv: Optional[Path] = None
    sample_submission_csv: Optional[Path] = None
    
    # Model
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    freeze_backbone: bool = True
    image_size: int = 224
    embedding_dim: int = 1280  # EfficientNet-B0 output
    
    # Training
    n_folds: int = 5
    epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    device: str = "cuda"
    seed: int = 42
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Tabular features
    categorical_features: List[str] = field(default_factory=lambda: ["State", "Species"])
    continuous_features: List[str] = field(default_factory=lambda: ["Pre_GSHH_NDVI", "Height_Ave_cm"])
    
    # Target components (base components to predict)
    base_targets: List[str] = field(default_factory=lambda: ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g"])
    all_targets: List[str] = field(default_factory=lambda: [
        "Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"
    ])
    
    # Target weights for evaluation
    target_weights: dict = field(default_factory=lambda: {
        "Dry_Green_g": 0.1,
        "Dry_Dead_g": 0.1,
        "Dry_Clover_g": 0.1,
        "GDM_g": 0.2,
        "Dry_Total_g": 0.5,
    })
    
    def __post_init__(self):
        """Set default paths if not provided."""
        if self.train_csv is None:
            self.train_csv = self.data_dir / "train.csv"
        if self.test_csv is None:
            self.test_csv = self.data_dir / "test.csv"
        if self.sample_submission_csv is None:
            self.sample_submission_csv = self.data_dir / "sample_submission.csv"
        
        # Ensure paths are Path objects
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.train_csv = Path(self.train_csv)
        self.test_csv = Path(self.test_csv)
        self.sample_submission_csv = Path(self.sample_submission_csv)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
