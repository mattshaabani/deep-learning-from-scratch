"""
src/phase3_rnn/sequence_ablation.py

Sequence-length ablation study: trains VanillaRNN and LSTM at
multiple sequence lengths, logging every run to MLflow -- the exact
same structured-grid pattern as Phase 2's depth_ablation.py, just
with sequence length substituted for network depth.

Usage:
    from src.phase3_rnn.sequence_ablation import run_sequence_ablation
    results = run_sequence_ablation(seq_lengths=[20, 50, 100, 200])
"""

import os
import json
import tempfile
from pathlib import Path
import mlflow
import torch
from src.phase3_rnn.rnn_cell import VanillaRNN
from src.phase3_rnn.lstm_cell import LSTM
from src.phase3_rnn.bptt_trainer import BPTTTrainer
from src.phase3_rnn.sequence_data import CharDataset
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

ARCHITECTURES = {
    "VanillaRNN": VanillaRNN,
    "LSTM":       LSTM,
}


def run_sequence_ablation(
    seq_lengths: list[int] = None,
    epochs: int = 5,
    steps_per_epoch: int = 30,
    experiment_name: str = "phase3-sequence-ablation",
) -> dict:
    """
    Run a full architecture x sequence_length grid, logging every
    run to MLflow, including per-timestep gradient norms as artifacts.
    """
    seq_lengths = seq_lengths or settings.phase3_sequence.seq_lengths_to_test

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    dataset = CharDataset()
    tmp_dir = Path(tempfile.gettempdir())

    all_results = {}

    for arch_name, arch_cls in ARCHITECTURES.items():
        for seq_length in seq_lengths:
            run_name = f"{arch_name}_seq{seq_length}"
            logger.info(f"Starting run: {run_name}")

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("architecture", arch_name)
                mlflow.log_param("seq_length", seq_length)
                mlflow.log_param("epochs", epochs)

                torch.manual_seed(42)
                model = arch_cls(
                    vocab_size=dataset.vocab_size,
                    embedding_size=settings.phase3_model.embedding_size,
                    hidden_size=settings.phase3_model.hidden_size,
                )

                # Apply the forget-gate-bias lesson from our investigation:
                # for LSTM, ensure the forget gate starts strongly biased
                # toward "remember" (f_t close to 1), consistent with what
                # we proved is necessary for gradient preservation
                if arch_name == "LSTM":
                    with torch.no_grad():
                        model.cell.forget_gate.bias.fill_(2.0)   # f_t ~= 0.88 at init
                    mlflow.log_param("forget_gate_bias_init", 2.0)

                trainer = BPTTTrainer(model, dataset, device="cpu")
                history = trainer.fit(
                    epochs=epochs, seq_length=seq_length,
                    steps_per_epoch=steps_per_epoch, verbose=False,
                )

                for epoch_idx in range(epochs):
                    mlflow.log_metrics({
                        "train_loss": history["train_loss"][epoch_idx],
                        "val_loss":   history["val_loss"][epoch_idx],
                    }, step=epoch_idx)

                mlflow.log_metric("final_val_loss", history["val_loss"][-1])

                grad_norms = history["grad_norms_last_step"]
                grad_norms_path = tmp_dir / f"{run_name}_grad_norms.json"
                with open(grad_norms_path, "w") as f:
                    json.dump(grad_norms, f, indent=2)
                mlflow.log_artifact(str(grad_norms_path))

                # "Vanishing severity" metric: earliest / latest timestep gradient ratio
                if len(grad_norms) >= 2:
                    ratio = grad_norms[0] / (grad_norms[-1] + 1e-12)
                    mlflow.log_metric("earliest_to_latest_grad_ratio", ratio)

                all_results[run_name] = history

                logger.info(f"Run complete: {run_name}", extra={
                    "final_val_loss": round(history["val_loss"][-1], 4),
                })

    return all_results