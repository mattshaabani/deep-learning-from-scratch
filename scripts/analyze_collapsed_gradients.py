import os
import json
import mlflow
import matplotlib.pyplot as plt

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = mlflow.tracking.MlflowClient()
experiment = mlflow.get_experiment_by_name("phase2-depth-ablation")

# Order by start_time DESCENDING so the most recent run for each
# name is encountered FIRST when we build the lookup dict
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
)

runs_by_name = {}
for r in runs:
    name = r.data.tags.get("mlflow.runName")
    if name not in runs_by_name:   # only keep the FIRST (=most recent) occurrence
        runs_by_name[name] = r

target_runs = [
    "PlainCNN_depth4",
    "PlainCNN_depth12",
    "PlainCNN_depth20",
    "ResNetCNN_depth12",
    "ResNetCNN_depth20",
]

fig, ax = plt.subplots(figsize=(11, 6))

for run_name in target_runs:
    run = runs_by_name.get(run_name)
    if run is None:
        print(f"Run not found: {run_name}")
        continue

    run_id = run.info.run_id
    artifacts = client.list_artifacts(run_id)
    print(f"\n{run_name} (run_id={run_id}): artifacts = {[a.path for a in artifacts]}")

    artifact_path = f"{run_name}_grad_norms.json"
    local_path = client.download_artifacts(run_id, artifact_path)
    with open(local_path, "r") as f:
        grad_norms = json.load(f)

    layer_indices = list(range(len(grad_norms)))
    norms = list(grad_norms.values())

    print(f"{run_name} gradient norms:")
    for layer_name, norm in grad_norms.items():
        print(f"  {layer_name}: {norm:.8f}")

    style = "--" if "PlainCNN" in run_name and "depth4" not in run_name else "-"
    ax.plot(layer_indices, norms, marker="o", linestyle=style, label=run_name)

ax.set_xlabel("Layer index (0 = closest to input)")
ax.set_ylabel("Gradient L2 norm (last training batch, final epoch)")
ax.set_yscale("log")
ax.set_title("Per-Layer Gradient Norms: Collapsed vs Healthy Networks")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("data/gradient_norms_collapse_analysis.png", dpi=150)
plt.show()

print("\nSaved plot to data/gradient_norms_collapse_analysis.png")