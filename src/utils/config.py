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


settings = Settings()