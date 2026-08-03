from src.phase2_cnn.depth_ablation import run_depth_ablation

results = run_depth_ablation(depths=[4, 8, 12, 16, 20], epochs=10)

print("\n=== FULL DEPTH ABLATION RESULTS ===")
for run_name, history in sorted(results.items()):
    print(f"{run_name:20s}: final_val_acc={history['val_acc'][-1]:.4f}, final_val_loss={history['val_loss'][-1]:.4f}")

print("\nRun 'mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db' to inspect all 10 runs.")