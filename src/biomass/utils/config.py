"""Configuration management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class Config:
    """Configuration for biomass prediction pipeline.
    
    Can be initialized from default values or loaded from a YAML file.
    """
    
    # Experiment
    experiment_name: str = "default"
    
    # Paths
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    train_csv: Optional[Path] = None
    test_csv: Optional[Path] = None
    sample_submission_csv: Optional[Path] = None
    
    # Model architecture
    model_name: str = "simple_concat"  # simple_concat, multimodal_attention
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    freeze_backbone: bool = True
    image_size: int = 224
    embedding_dim: int = 1280  # EfficientNet-B0 output
    
    # Model hyperparameters
    hidden_dim: int = 256
    dropout: float = 0.3
    categorical_embedding_dim: int = 16
    
    # Advanced model options (for multimodal_attention)
    use_cross_attention: bool = True
    use_transformer_blocks: bool = True
    num_transformer_layers: int = 2
    
    # Training
    n_folds: int = 5
    epochs: int = 30
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
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'Config':
        """Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Config object with values from YAML
            
        Example:
            >>> config = Config.from_yaml('configs/train_simple_concat.yaml')
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Create config from YAML
        config = cls()
        
        # Parse nested structure
        if 'experiment_name' in yaml_config:
            config.experiment_name = yaml_config['experiment_name']
        
        if 'output_dir' in yaml_config:
            config.output_dir = Path(yaml_config['output_dir'])
        
        # Data paths
        if 'data' in yaml_config:
            data = yaml_config['data']
            if 'data_dir' in data:
                config.data_dir = Path(data['data_dir'])
            if 'train_csv' in data:
                config.train_csv = Path(data['train_csv'])
            if 'test_csv' in data:
                config.test_csv = Path(data['test_csv'])
            if 'sample_submission_csv' in data:
                config.sample_submission_csv = Path(data['sample_submission_csv'])
        
        # Model configuration
        if 'model' in yaml_config:
            model = yaml_config['model']
            if 'name' in model:
                config.model_name = model['name']
            if 'backbone' in model:
                config.backbone = model['backbone']
            if 'pretrained' in model:
                config.pretrained = model['pretrained']
            if 'freeze_backbone' in model:
                config.freeze_backbone = model['freeze_backbone']
            if 'hidden_dim' in model:
                config.hidden_dim = model['hidden_dim']
            if 'dropout' in model:
                config.dropout = model['dropout']
            if 'categorical_embedding_dim' in model:
                config.categorical_embedding_dim = model['categorical_embedding_dim']
            
            # Advanced options
            if 'use_cross_attention' in model:
                config.use_cross_attention = model['use_cross_attention']
            if 'use_transformer_blocks' in model:
                config.use_transformer_blocks = model['use_transformer_blocks']
            if 'num_transformer_layers' in model:
                config.num_transformer_layers = model['num_transformer_layers']
        
        # Image configuration
        if 'image' in yaml_config:
            image = yaml_config['image']
            if 'size' in image:
                config.image_size = image['size']
        
        # Training configuration
        if 'training' in yaml_config:
            training = yaml_config['training']
            if 'n_folds' in training:
                config.n_folds = training['n_folds']
            if 'epochs' in training:
                config.epochs = training['epochs']
            if 'batch_size' in training:
                config.batch_size = training['batch_size']
            if 'learning_rate' in training:
                config.learning_rate = training['learning_rate']
            if 'weight_decay' in training:
                config.weight_decay = training['weight_decay']
            if 'num_workers' in training:
                config.num_workers = training['num_workers']
            if 'use_amp' in training:
                config.use_amp = training['use_amp']
            if 'seed' in training:
                config.seed = training['seed']
        
        # Features
        if 'features' in yaml_config:
            features = yaml_config['features']
            if 'categorical' in features:
                config.categorical_features = features['categorical']
            if 'continuous' in features:
                config.continuous_features = features['continuous']
        
        # Targets
        if 'targets' in yaml_config:
            targets = yaml_config['targets']
            if 'base' in targets:
                config.base_targets = targets['base']
            if 'all' in targets:
                config.all_targets = targets['all']
            if 'weights' in targets:
                config.target_weights = targets['weights']
        
        # Run post_init to set up paths
        config.__post_init__()
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return {
            'experiment_name': self.experiment_name,
            'data': {
                'data_dir': str(self.data_dir),
                'train_csv': str(self.train_csv) if self.train_csv else None,
                'test_csv': str(self.test_csv) if self.test_csv else None,
                'sample_submission_csv': str(self.sample_submission_csv) if self.sample_submission_csv else None,
            },
            'model': {
                'name': self.model_name,
                'backbone': self.backbone,
                'pretrained': self.pretrained,
                'freeze_backbone': self.freeze_backbone,
                'hidden_dim': self.hidden_dim,
                'dropout': self.dropout,
                'categorical_embedding_dim': self.categorical_embedding_dim,
                'use_cross_attention': self.use_cross_attention,
                'use_transformer_blocks': self.use_transformer_blocks,
                'num_transformer_layers': self.num_transformer_layers,
            },
            'image': {
                'size': self.image_size,
            },
            'training': {
                'n_folds': self.n_folds,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'num_workers': self.num_workers,
                'use_amp': self.use_amp,
                'seed': self.seed,
            },
            'features': {
                'categorical': self.categorical_features,
                'continuous': self.continuous_features,
            },
            'targets': {
                'base': self.base_targets,
                'all': self.all_targets,
                'weights': self.target_weights,
            },
        }
    
    def save_yaml(self, yaml_path: str | Path):
        """Save configuration to YAML file.
        
        Args:
            yaml_path: Path where to save YAML config
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
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
