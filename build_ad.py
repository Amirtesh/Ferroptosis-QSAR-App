# Run this once in your training environment
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Load training SMILES (final curated dataset, 1,609 compounds)
df = pd.read_csv("ferroptosis_features.csv")  # must have 'SMILES' column

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def get_fp_array(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = generator.GetFingerprint(mol)
    return np.array(list(fp), dtype=np.uint8)

train_fps = []
for smi in df["SMILES"]:
    fp = get_fp_array(smi)
    if fp is not None:
        train_fps.append(fp)

train_fps = np.array(train_fps, dtype=np.uint8)
print(f"Training fingerprints built: {train_fps.shape[0]}")

np.save("ad_training_matrix.npy", train_fps)
print("Saved: ad_training_matrix.npy")
