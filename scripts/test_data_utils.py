from src.common.data_utils import load_moons, load_breast_cancer_data, KFoldSplitter, train_test_split_manual

print("=== Loading moons dataset ===")
X_moons, y_moons = load_moons()
print(f"X shape: {X_moons.shape}, y shape: {y_moons.shape}")
print(f"Class balance: {y_moons.mean():.2%} class 1")

print("\n=== Loading breast cancer dataset ===")
X_bc, y_bc = load_breast_cancer_data()
print(f"X shape: {X_bc.shape}, y shape: {y_bc.shape}")
print(f"Feature means (should be ~0 after scaling): {X_bc.mean(axis=0)[:5].round(4)}")
print(f"Feature stds (should be ~1 after scaling): {X_bc.std(axis=0)[:5].round(4)}")

print("\n=== Testing manual train/test split ===")
X_train, X_test, y_train, y_test = train_test_split_manual(X_bc, y_bc, test_size=0.2)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

print("\n=== Testing manual K-Fold splitter ===")
splitter = KFoldSplitter(k_folds=5)
for i, (train_idx, val_idx) in enumerate(splitter.split(X_bc)):
    print(f"  Fold {i+1}: train={len(train_idx)}, val={len(val_idx)}, "
          f"overlap={len(set(train_idx) & set(val_idx))} (should be 0)")