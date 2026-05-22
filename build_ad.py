# Run this once in your training environment
import numpy as np
import joblib
import json
from predict import build_feature_vector, load_feature_info
import pandas as pd

# Load your training SMILES (the full 1624 compounds)
df = pd.read_csv("ferroptosis_features.csv")  # must have 'SMILES' column

feature_columns, n_features, _ = load_feature_info("feature_info.json")

X_train = []
for smi in df["SMILES"]:
    x = build_feature_vector(smi, feature_columns, n_features)
    if x is not None:
        X_train.append(x)

X_train = np.array(X_train)

# Save mean vector and std for Euclidean AD
ad_data = {
    "mean": X_train.mean(axis=0).tolist(),
    "std": X_train.std(axis=0).tolist(),
    "threshold": None  # will set below
}

# Compute per-compound distances to centroid in training set
dists = np.linalg.norm((X_train - X_train.mean(axis=0)) / (X_train.std(axis=0) + 1e-9), axis=1)
ad_data["threshold"] = float(np.percentile(dists, 95))  # 95th percentile = AD boundary

np.save("ad_training_matrix.npy", X_train)
with open("ad_stats.json", "w") as f:
    json.dump(ad_data, f)

print("Saved: ad_training_matrix.npy, ad_stats.json")
