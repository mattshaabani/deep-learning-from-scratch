from src.phase2_cnn.depth_ablation import run_depth_ablation

# Small smoke test first: 2 depths, 3 epochs, to confirm MLflow logging works
# before committing to the full expensive grid
results = run_depth_ablation(depths=[4, 12], epochs=3)

print("\n=== SMOKE TEST RESULTS ===")
for run_name, history in results.items():
    print(f"{run_name}: final_val_acc={history['val_acc'][-1]:.4f}")

print("\nRun 'mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db' to inspect the logged runs.")