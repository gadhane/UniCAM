from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data/PetImages"
DEFAULT_MODELS_DIR = BASE_DIR / "models"
DEFAULT_LOGS_DIR = BASE_DIR / "PetImageLogs"
DEFAULT_RESULTS_DIR = BASE_DIR / "RemoveResults"
DEFAULT_CSV_DIR = BASE_DIR / "csv"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PET_CLASSES = ("Cat", "Dog")


def seed_everything(seed: int = 5) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_dir(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    return Path(path).expanduser().resolve()


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            ),
            transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224, normalize: bool = True) -> transforms.Compose:
    steps = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    if normalize:
        steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(steps)


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def load_split_dataset(
    data_dir: str | Path,
    split: str,
    transform: Callable,
):
    root = resolve_dir(data_dir, DEFAULT_DATA_DIR)
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Expected split directory '{split_dir}' to exist."
        )
    return datasets.ImageFolder(split_dir, transform)


def build_resnet(model_name: str, num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    builders = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
    }
    weights_enums = {
        "resnet18": getattr(models, "ResNet18_Weights", None),
        "resnet34": getattr(models, "ResNet34_Weights", None),
        "resnet50": getattr(models, "ResNet50_Weights", None),
        "resnet101": getattr(models, "ResNet101_Weights", None),
    }
    if model_name not in builders:
        raise ValueError(f"Unsupported model '{model_name}'.")

    builder = builders[model_name]
    try:
        weights = None
        weights_enum = weights_enums[model_name]
        if pretrained and weights_enum is None:
            raise TypeError
        if pretrained and weights_enum is not None:
            weights = weights_enum.DEFAULT
        model = builder(weights=weights)
    except TypeError:
        model = builder(pretrained=pretrained)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> nn.Module:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: '{checkpoint_path}'")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def get_resnet_target_layer(model: nn.Module, layer_name: str = "layer4") -> nn.Module:
    if not hasattr(model, layer_name):
        raise ValueError(f"Model does not expose layer '{layer_name}'.")
    layer = getattr(model, layer_name)
    return layer[-1]
