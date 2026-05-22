#!/usr/bin/env python3
import argparse
import json
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdFingerprintGenerator
import joblib

def calculate_molecular_descriptors(mol):
    d = {}
    d["MW"] = Descriptors.MolWt(mol)
    d["LogP"] = Descriptors.MolLogP(mol)
    d["TPSA"] = Descriptors.TPSA(mol)
    d["HBD"] = Lipinski.NumHDonors(mol)
    d["HBA"] = Lipinski.NumHAcceptors(mol)
    d["RotatableBonds"] = Lipinski.NumRotatableBonds(mol)

    d["NumAtoms"] = mol.GetNumAtoms()
    d["NumHeavyAtoms"] = Lipinski.HeavyAtomCount(mol)
    d["NumHeteroatoms"] = Lipinski.NumHeteroatoms(mol)
    d["NumBonds"] = mol.GetNumBonds()
    d["NumAromaticRings"] = Lipinski.NumAromaticRings(mol)
    d["NumSaturatedRings"] = Lipinski.NumSaturatedRings(mol)
    d["NumAliphaticRings"] = Lipinski.NumAliphaticRings(mol)

    d["RingCount"] = Descriptors.RingCount(mol)
    d["NumAromaticCarbocycles"] = Lipinski.NumAromaticCarbocycles(mol)
    d["NumAromaticHeterocycles"] = Lipinski.NumAromaticHeterocycles(mol)

    d["BertzCT"] = Descriptors.BertzCT(mol)
    d["Chi0v"] = Descriptors.Chi0v(mol)
    d["Chi1v"] = Descriptors.Chi1v(mol)
    d["Kappa1"] = Descriptors.Kappa1(mol)
    d["Kappa2"] = Descriptors.Kappa2(mol)
    d["Kappa3"] = Descriptors.Kappa3(mol)

    d["MolMR"] = Crippen.MolMR(mol)
    d["PEOE_VSA1"] = Descriptors.PEOE_VSA1(mol)
    d["PEOE_VSA2"] = Descriptors.PEOE_VSA2(mol)
    d["SMR_VSA1"] = Descriptors.SMR_VSA1(mol)
    d["SMR_VSA5"] = Descriptors.SMR_VSA5(mol)
    d["SlogP_VSA1"] = Descriptors.SlogP_VSA1(mol)
    d["SlogP_VSA2"] = Descriptors.SlogP_VSA2(mol)

    d["MaxPartialCharge"] = Descriptors.MaxPartialCharge(mol)
    d["MinPartialCharge"] = Descriptors.MinPartialCharge(mol)
    d["MaxAbsPartialCharge"] = Descriptors.MaxAbsPartialCharge(mol)

    d["EState_VSA1"] = Descriptors.EState_VSA1(mol)
    d["EState_VSA2"] = Descriptors.EState_VSA2(mol)
    d["VSA_EState1"] = Descriptors.VSA_EState1(mol)
    d["VSA_EState2"] = Descriptors.VSA_EState2(mol)

    d["FractionCsp3"] = Lipinski.FractionCSP3(mol)
    d["LabuteASA"] = Descriptors.LabuteASA(mol)
    d["BalabanJ"] = Descriptors.BalabanJ(mol)
    d["HallKierAlpha"] = Descriptors.HallKierAlpha(mol)

    d["MolLogP_Crippen"] = Crippen.MolLogP(mol)
    d["NumValenceElectrons"] = Descriptors.NumValenceElectrons(mol)
    d["NumRadicalElectrons"] = Descriptors.NumRadicalElectrons(mol)

    d["FlexibilityIndex"] = d["RotatableBonds"] / max(d["NumBonds"], 1)
    return d

def generate_morgan_fp(mol, radius=2, nbits=2048):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fp = gen.GetFingerprint(mol)
    return np.array(list(fp), dtype=np.float64)

def load_feature_info(feature_info_path):
    with open(feature_info_path, "r") as f:
        info = json.load(f)
    cols = info["feature_columns"]
    n_features = int(info.get("n_features", len(cols)))
    return cols, n_features, info.get("target_encoding", {})

def build_feature_vector(smiles, feature_columns, n_features, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    d = calculate_molecular_descriptors(mol)
    fp = generate_morgan_fp(mol, radius=radius, nbits=nbits)

    idx_map = {c: i for i, c in enumerate(feature_columns)}
    x = np.zeros(n_features, dtype=np.float64)

    for k, v in d.items():
        j = idx_map.get(k, None)
        if j is not None:
            x[j] = float(v)

    for i in range(nbits):
        j = idx_map.get(f"MORGANbit{i}", None)
        if j is not None:
            x[j] = fp[i]

    return x

def read_smiles_input(smiles_arg=None, smiles_file=None):
    if smiles_arg is not None:
        return [smiles_arg.strip()]
    out = []
    with open(smiles_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ensemble_model.pkl")
    ap.add_argument("--feature_info", default="feature_info.json")
    ap.add_argument("--smiles", default=None)
    ap.add_argument("--smiles_file", default=None)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    if (args.smiles is None) == (args.smiles_file is None):
        print("Provide exactly one: --smiles OR --smiles_file", file=sys.stderr)
        sys.exit(1)

    feature_columns, n_features, target_encoding = load_feature_info(args.feature_info)
    model = joblib.load(args.model)

    smiles_list = read_smiles_input(args.smiles, args.smiles_file)
    if len(smiles_list) == 0:
        print("No SMILES found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for smi in smiles_list:
        x = build_feature_vector(smi, feature_columns, n_features, radius=2, nbits=2048)
        if x is None:
            rows.append({"SMILES": smi, "Prediction": "INVALID_SMILES", "Probability": ""})
            continue

        proba = model.predict_proba(x.reshape(1, -1))[0]
        idx = int(np.argmax(proba))
        p = float(proba[idx])
        pred_label = "Ferroptosis-Inducer" if idx == 1 else "Ferroptosis-Inhibitor"
        rows.append({"SMILES": smi, "Prediction": pred_label, "Probability": p})

    out_df = pd.DataFrame(rows, columns=["SMILES", "Prediction", "Probability"])
    out_df.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")

if __name__ == "__main__":
    main()

