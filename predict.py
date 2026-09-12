#!/usr/bin/env python3
"""Ferroptosis prediction with closest-neighbour Tanimoto AD assessment."""

import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator

DEFAULT_MODEL = "ensemble_model.pkl"
DEFAULT_FEATURE_INFO = "feature_info.json"
DEFAULT_AD_MATRIX = "ad_training_matrix.npy"
DEFAULT_OUTPUT = "predictions.csv"
FP_RADIUS = 2
FP_NBITS = 2048
AD_DISTANCE_THRESHOLD = 0.7
GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)


def calculate_molecular_descriptors(mol):
    d = {"MW": Descriptors.MolWt(mol), "LogP": Descriptors.MolLogP(mol), "TPSA": Descriptors.TPSA(mol), "HBD": Lipinski.NumHDonors(mol), "HBA": Lipinski.NumHAcceptors(mol), "RotatableBonds": Lipinski.NumRotatableBonds(mol), "NumAtoms": mol.GetNumAtoms(), "NumHeavyAtoms": Lipinski.HeavyAtomCount(mol), "NumHeteroatoms": Lipinski.NumHeteroatoms(mol), "NumBonds": mol.GetNumBonds(), "NumAromaticRings": Lipinski.NumAromaticRings(mol), "NumSaturatedRings": Lipinski.NumSaturatedRings(mol), "NumAliphaticRings": Lipinski.NumAliphaticRings(mol), "RingCount": Descriptors.RingCount(mol), "NumAromaticCarbocycles": Lipinski.NumAromaticCarbocycles(mol), "NumAromaticHeterocycles": Lipinski.NumAromaticHeterocycles(mol), "BertzCT": Descriptors.BertzCT(mol), "Chi0v": Descriptors.Chi0v(mol), "Chi1v": Descriptors.Chi1v(mol), "Kappa1": Descriptors.Kappa1(mol), "Kappa2": Descriptors.Kappa2(mol), "Kappa3": Descriptors.Kappa3(mol), "MolMR": Crippen.MolMR(mol), "PEOE_VSA1": Descriptors.PEOE_VSA1(mol), "PEOE_VSA2": Descriptors.PEOE_VSA2(mol), "SMR_VSA1": Descriptors.SMR_VSA1(mol), "SMR_VSA5": Descriptors.SMR_VSA5(mol), "SlogP_VSA1": Descriptors.SlogP_VSA1(mol), "SlogP_VSA2": Descriptors.SlogP_VSA2(mol), "MaxPartialCharge": Descriptors.MaxPartialCharge(mol), "MinPartialCharge": Descriptors.MinPartialCharge(mol), "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge(mol), "EState_VSA1": Descriptors.EState_VSA1(mol), "EState_VSA2": Descriptors.EState_VSA2(mol), "VSA_EState1": Descriptors.VSA_EState1(mol), "VSA_EState2": Descriptors.VSA_EState2(mol), "FractionCsp3": Lipinski.FractionCSP3(mol), "LabuteASA": Descriptors.LabuteASA(mol), "BalabanJ": Descriptors.BalabanJ(mol), "HallKierAlpha": Descriptors.HallKierAlpha(mol), "MolLogP_Crippen": Crippen.MolLogP(mol), "NumValenceElectrons": Descriptors.NumValenceElectrons(mol), "NumRadicalElectrons": Descriptors.NumRadicalElectrons(mol)}
    d["FlexibilityIndex"] = d["RotatableBonds"] / max(d["NumBonds"], 1)
    return d


def generate_fingerprint(mol):
    return GENERATOR.GetFingerprint(mol)


def load_feature_info(path):
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    columns = info["feature_columns"]
    return columns, int(info.get("n_features", len(columns)))


def build_feature_vector(smiles, feature_columns, n_features):
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None, None
    descriptors = calculate_molecular_descriptors(mol)
    fingerprint = generate_fingerprint(mol)
    index = {name: i for i, name in enumerate(feature_columns)}
    vector = np.zeros(n_features, dtype=np.float64)
    for name, value in descriptors.items():
        if name in index:
            vector[index[name]] = float(value)
    for bit in range(FP_NBITS):
        for name in (f"MORGAN_bit_{bit}", f"MORGANbit{bit}"):
            if name in index:
                vector[index[name]] = float(fingerprint.GetBit(bit))
                break
    return vector, fingerprint


def load_ad_matrix():
    matrix = np.load(DEFAULT_AD_MATRIX, allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[1] != FP_NBITS:
        raise ValueError(f"{DEFAULT_AD_MATRIX} must have shape (n, {FP_NBITS}); got {matrix.shape}")
    return matrix.astype(np.uint8, copy=False)


def array_to_bitvect(row):
    bitvect = DataStructs.ExplicitBitVect(FP_NBITS)
    for bit in np.flatnonzero(row):
        bitvect.SetBit(int(bit))
    return bitvect


def compute_ad(query_fingerprint, training_matrix):
    training_fingerprints = [array_to_bitvect(row) for row in training_matrix]
    similarities = DataStructs.BulkTanimotoSimilarity(query_fingerprint, training_fingerprints)
    closest_distance = 1.0 - float(max(similarities))
    return round(closest_distance, 4), bool(closest_distance <= AD_DISTANCE_THRESHOLD)


def read_smiles(smiles=None, smiles_file=None):
    if smiles is not None:
        return [smiles.strip()]
    if smiles_file is None:
        raise ValueError("Provide --smiles or --smiles_file.")
    with open(smiles_file, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--feature_info", default=DEFAULT_FEATURE_INFO)
    parser.add_argument("--smiles")
    parser.add_argument("--smiles_file")
    parser.add_argument("--out", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if (args.smiles is None) == (args.smiles_file is None):
        raise ValueError("Provide exactly one of --smiles or --smiles_file.")

    feature_columns, n_features = load_feature_info(args.feature_info)
    model = joblib.load(args.model)
    ad_matrix = load_ad_matrix()
    rows = []

    for smiles in read_smiles(args.smiles, args.smiles_file):
        vector, fingerprint = build_feature_vector(smiles, feature_columns, n_features)
        if vector is None:
            rows.append({"SMILES": smiles, "Prediction": "INVALID_SMILES", "Probability": np.nan, "AD_ClosestDistance": np.nan, "AD_WithinDomain": np.nan})
            continue
        probabilities = model.predict_proba(vector.reshape(1, -1))[0]
        predicted_index = int(np.argmax(probabilities))
        prediction = "Ferroptosis-Inducer" if predicted_index == 1 else "Ferroptosis-Inhibitor"
        distance, within_domain = compute_ad(fingerprint, ad_matrix)
        rows.append({"SMILES": smiles, "Prediction": prediction, "Probability": round(float(probabilities[predicted_index]), 4), "AD_ClosestDistance": distance, "AD_WithinDomain": within_domain})

    output = pd.DataFrame(rows, columns=["SMILES", "Prediction", "Probability", "AD_ClosestDistance", "AD_WithinDomain"])
    output.to_csv(args.out, index=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
