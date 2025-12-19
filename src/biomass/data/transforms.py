"""Image transforms and preprocessing (Step 4)."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_transforms(image_size: int = 224):
    """Get training image transforms with augmentation.
    
    Implements mild augmentations suitable for pasture images:
    - Geometric: resize, random crop, horizontal flip
    - Color: mild brightness/contrast jitter
    - Normalization: ImageNet stats (for pre-trained backbones)
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(image_size + 32, image_size + 32),
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0),
        ], p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224):
    """Get validation/test image transforms (no augmentation).
    
    Args:
        image_size: Target image size
        
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def load_image(image_path: str):
    """Load and convert image to RGB.
    
    Args:
        image_path: Path to image file
        
    Returns:
        RGB image as numpy array
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
