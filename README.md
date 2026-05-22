# FerroptosisPredictor

A QSAR-based web application for classifying small molecules as ferroptosis inducers or ferroptosis inhibitors, built as a companion tool for the associated manuscript submitted to the *Journal of Cheminformatics*.

---

## Overview

FerroptosisPredictor provides an interactive interface for running predictions through a trained soft-voting ensemble classifier (Random Forest + XGBoost). The model was trained on a curated dataset of 1,624 ferroptosis modulators and uses 2,092 molecular features derived from physicochemical descriptors and Morgan fingerprints.

The application supports single-compound analysis with full interpretability output (SHAP feature contributions, 2D structure rendering, functional group flags, and applicability domain assessment), as well as batch processing for multiple compounds via CSV upload or direct SMILES paste.

---

## Features

- **Single SMILES prediction**: Paste any valid SMILES string and receive a class prediction (Inducer / Inhibitor), class probabilities, applicability domain status, a SHAP-based feature importance chart, and a functional group analysis table.
- **Batch prediction**: Upload a CSV file (requires a `SMILES` column; `Name` column optional) or paste multiple SMILES strings (one per line, optional name after whitespace). Results are displayed in a downloadable table with summary charts.
- **Applicability domain**: Normalized Euclidean distance from the training set centroid is computed; compounds exceeding the 95th-percentile threshold of the training distribution are flagged.
- **SHAP explanations**: Weighted ensemble SHAP values (XGBoost weight = 2, Random Forest weight = 1) are computed with respect to the Inducer class. Positive values push toward Inducer; negative values push toward Inhibitor.
- **Functional group analysis**: Substructure matching for four pharmacophoric motifs with class tendency annotations derived from the paper's enrichment analysis.

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Soft-voting ensemble (Random Forest + XGBoost) |
| Ensemble weights | XGBoost: 2, Random Forest: 1 |
| Training set size | 1,624 ferroptosis modulators |
| Feature set | 44 physicochemical descriptors + 2,048-bit Morgan fingerprints (radius 2) |
| Total features | 2,092 |
| Applicability domain | Normalized L2 distance; threshold at 95th percentile of training distances |

Label encoding: 0 = Ferroptosis Inhibitor, 1 = Ferroptosis Inducer.

---

## Repository Structure

```
.
├── app.py                    # Streamlit application
├── predict.py                # Feature computation pipeline
├── ensemble_model.pkl        # Trained ensemble model (joblib)
├── feature_info.json         # Ordered feature column list and metadata
├── ad_stats.json             # Applicability domain mean, std, and threshold
├── ad_training_matrix.npy    # Training feature matrix for AD computation
├── requirements.txt          # Python dependencies
├── packages.txt              # System packages for Streamlit Cloud
└── runtime.txt               # Python version pin for Streamlit Cloud
```

---

## Running Locally

**Prerequisites**: Python 3.11, a virtual environment with the dependencies below.

```bash
# Clone the repository
git clone https://github.com/Amirtesh/Ferroptosis-QSAR-App.git
cd Ferroptosis-QSAR-App

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## Deployment

The application is configured for deployment on [Streamlit Community Cloud](https://streamlit.io/cloud). The `packages.txt` file ensures the system-level dependency required by RDKit's 2D drawing module (`libxrender1`) is installed automatically.

Live app: *(add Streamlit Cloud URL here after deployment)*

---

## Input Format

**Single SMILES**: Any valid SMILES string accepted by RDKit. Invalid strings are caught and flagged with a warning.

**Batch CSV**: Must contain a column named `SMILES`. An optional `Name` column is used as the row identifier in results. Example:

```
Name,SMILES
Erastin,C(=O)(c1cc(OC)c(OC)cc1)Nc1ccc(cc1)N1CCN(CC1)C(=O)c1ccccn1
RSL3,O=C(OCC(=O)c1ccc(Cl)cc1)c1cccc([C@@H]2CCCCN2Cc2ccccc2)c1
```

**Batch paste**: One SMILES per line. An optional name may follow the SMILES separated by whitespace. Lines beginning with `#` are treated as comments.

```
CC(=O)Oc1ccccc1C(=O)O  Aspirin
C1=CC=CC=C1
# This line is ignored
```

---

## Dependencies

```
streamlit >= 1.32.0
rdkit
scikit-learn
xgboost
shap
matplotlib
numpy
pandas
joblib
```

---

## Citation

If you use this tool in your research, please cite the associated manuscript:

> *(Citation will be added upon publication)*

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Disclaimer

This tool is intended for research purposes only. Predictions are model-derived and should not be used as a substitute for experimental validation. The applicability domain check provides an indication of prediction reliability but does not guarantee accuracy for compounds outside the training distribution.
