"""
src/phase2_cnn/depth_ablation.py

Structured depth ablation study: trains PlainCNN and ResNetCNN at
multiple depths, logging every run to MLflow for professional,
queryable experiment comparison.
"""

import os
import json
import tempfile
from pathlib import Path
import mlflow
import numpy as np
from src.phase2_cnn.cnn_architectures import PlainCNN, ResNetCNN, count_parameters
from src.phase2_cnn.trainer import CNNTrainer
from src.phase2_cnn.data_augmentation import get_cifar10_loaders
from src.utils.logger import get_logger

logger = get_logger(__name__)

ARCHITECTURES = {
    "PlainCNN":  PlainCNN,
    "ResNetCNN": ResNetCNN,
}


def run_depth_ablation(
    depths: list[int] = None,
    epochs: int = 10,
    experiment_name: str = "phase2-depth-ablation",
) -> dict:
    depths = depths or [4, 8, 12, 16, 20]

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    train_loader, val_loader, test_loader = get_cifar10_loaders(use_augmentation=True)

    all_results = {}
    tmp_dir = Path(tempfile.gettempdir())

    for arch_name, arch_cls in ARCHITECTURES.items():
        for depth in depths:
            run_name = f"{arch_name}_depth{depth}"
            logger.info(f"Starting run: {run_name}")

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("architecture", arch_name)
                mlflow.log_param("depth", depth)
                mlflow.log_param("epochs", epochs)

                model = arch_cls(num_classes=4, depth=depth)
                n_params = count_parameters(model)
                mlflow.log_param("trainable_params", n_params)

                trainer = CNNTrainer(model, device="cpu")
                history = trainer.fit(train_loader, val_loader, epochs=epochs, verbose=False)

                for epoch_idx in range(epochs):
                    mlflow.log_metrics({
                        "train_loss": history["train_loss"][epoch_idx],
                        "train_acc":  history["train_acc"][epoch_idx],
                        "val_loss":   history["val_loss"][epoch_idx],
                        "val_acc":    history["val_acc"][epoch_idx],
                    }, step=epoch_idx)

                mlflow.log_metrics({
                    "final_train_acc": history["train_acc"][-1],
                    "final_val_acc":   history["val_acc"][-1],
                    "final_val_loss":  history["val_loss"][-1],
                })

                final_grad_norms = history["grad_norms_by_epoch"][-1]
                grad_norms_path = tmp_dir / f"{run_name}_grad_norms.json"
                with open(grad_norms_path, "w") as f:
                    json.dump(final_grad_norms, f, indent=2)
                mlflow.log_artifact(str(grad_norms_path))

                norms = list(final_grad_norms.values())
                if len(norms) >= 2:
                    vanishing_ratio = norms[0] / (norms[-1] + 1e-10)
                    mlflow.log_metric("first_to_last_grad_ratio", vanishing_ratio)

                all_results[run_name] = history

                logger.info(f"Run complete: {run_name}", extra={
                    "final_val_acc": round(history["val_acc"][-1], 4),
                    "n_params": n_params,
                })

    return all_results