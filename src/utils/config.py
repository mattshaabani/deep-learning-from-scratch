"""
src/utils/config.py

Central configuration loader for the deep learning from scratch project.
"""

from pathlib import Path
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

ROOT_DIR = Path(__file__).parent.parent.parent


def load_yaml(filename: str) -> dict:
    path = ROOT_DIR / "configs" / filename
    with open(path, "r") as f:
        return yaml.safe_load(f)


_phase1_cfg = load_yaml("phase1_config.yaml")


class DatasetConfig:
    moons:         dict = _phase1_cfg["datasets"]["moons"]
    breast_cancer: dict = _phase1_cfg["datasets"]["breast_cancer"]


class NetworkConfig:
    hidden_layers:      list  = _phase1_cfg["network"]["hidden_layers"]
    activation:          str   = _phase1_cfg["network"]["activation"]
    output_activation:   str   = _phase1_cfg["network"]["output_activation"]
    weight_init:         str   = _phase1_cfg["network"]["weight_init"]


class TrainingConfig:
    epochs:         int   = _phase1_cfg["training"]["epochs"]
    learning_rate:  float = _phase1_cfg["training"]["learning_rate"]
    batch_size:     int   = _phase1_cfg["training"]["batch_size"]
    optimizer:      str   = _phase1_cfg["training"]["optimizer"]
    momentum_beta:  float = _phase1_cfg["training"]["momentum_beta"]
    rmsprop_beta:   float = _phase1_cfg["training"]["rmsprop_beta"]
    adam_beta1:     float = _phase1_cfg["training"]["adam_beta1"]
    adam_beta2:     float = _phase1_cfg["training"]["adam_beta2"]
    adam_epsilon:   float = _phase1_cfg["training"]["adam_epsilon"]


class RegularizationConfig:
    l1_lambda:                float = _phase1_cfg["regularization"]["l1_lambda"]
    l2_lambda:                float = _phase1_cfg["regularization"]["l2_lambda"]
    dropout_rate:              float = _phase1_cfg["regularization"]["dropout_rate"]
    early_stopping_patience:   int   = _phase1_cfg["regularization"]["early_stopping_patience"]


class CrossValidationConfig:
    k_folds:       int  = _phase1_cfg["cross_validation"]["k_folds"]
    shuffle:       bool = _phase1_cfg["cross_validation"]["shuffle"]
    random_state:  int  = _phase1_cfg["cross_validation"]["random_state"]


_phase2_cfg = load_yaml("phase2_config.yaml")

class Phase2DatasetConfig:
    name:                   str  = _phase2_cfg["dataset"]["name"]
    subset_classes:          int  = _phase2_cfg["dataset"]["subset_classes"]
    subset_size_per_class:    int  = _phase2_cfg["dataset"]["subset_size_per_class"]
    full_run_all_classes:      bool = _phase2_cfg["dataset"]["full_run_all_classes"]
    test_size:                  float = _phase2_cfg["dataset"]["test_size"]
    random_state:                 int  = _phase2_cfg["dataset"]["random_state"]

class Phase2ConvConfig:
    kernel_size: int = _phase2_cfg["conv"]["kernel_size"]
    stride:       int = _phase2_cfg["conv"]["stride"]
    padding:       int = _phase2_cfg["conv"]["padding"]

class Phase2TrainingConfig:
    epochs:         int   = _phase2_cfg["training"]["epochs"]
    learning_rate:   float = _phase2_cfg["training"]["learning_rate"]
    batch_size:       int   = _phase2_cfg["training"]["batch_size"]
    optimizer:         str   = _phase2_cfg["training"]["optimizer"]
    weight_decay:        float = _phase2_cfg["training"]["weight_decay"]

class Phase2AugmentationConfig:
    horizontal_flip:      bool = _phase2_cfg["augmentation"]["horizontal_flip"]
    rotation_degrees:       int  = _phase2_cfg["augmentation"]["rotation_degrees"]
    random_crop_padding:      int  = _phase2_cfg["augmentation"]["random_crop_padding"]


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    log_level: str = Field(default="INFO")


class Settings:
    dataset:           DatasetConfig          = DatasetConfig()
    network:            NetworkConfig          = NetworkConfig()
    training:            TrainingConfig         = TrainingConfig()
    regularization:      RegularizationConfig   = RegularizationConfig()
    cross_validation:    CrossValidationConfig  = CrossValidationConfig()
    env:                 EnvSettings            = EnvSettings()
    root_dir:            Path                   = ROOT_DIR
    phase2_dataset:     Phase2DatasetConfig     = Phase2DatasetConfig()
    phase2_conv:          Phase2ConvConfig        = Phase2ConvConfig()
    phase2_training:        Phase2TrainingConfig    = Phase2TrainingConfig()
    phase2_augmentation:       Phase2AugmentationConfig = Phase2AugmentationConfig()


settings = Settings()