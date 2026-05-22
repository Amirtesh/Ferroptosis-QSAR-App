import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors

# Suppress noisy warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ensemble_model.pkl")
FEATURE_INFO_PATH = os.path.join(BASE_DIR, "feature_info.json")
AD_STATS_PATH = os.path.join(BASE_DIR, "ad_stats.json")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="FerroptosisPredictor",
    page_icon=":microscope:",
)

# ---------------------------------------------------------------------------
# Imports from predict.py (local)
# ---------------------------------------------------------------------------
sys.path.insert(0, BASE_DIR)
from predict import build_feature_vector, load_feature_info

# ---------------------------------------------------------------------------
# Resource loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner="Loading feature info...")
def load_features():
    cols, n_feat, _ = load_feature_info(FEATURE_INFO_PATH)
    return cols, n_feat

AD_MATRIX_PATH = os.path.join(BASE_DIR, "ad_training_matrix.npy")

@st.cache_resource(show_spinner="Loading applicability domain stats...")
def load_ad_stats():
    with open(AD_STATS_PATH, "r") as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float64)
    std = np.array(stats["std"], dtype=np.float64)
    threshold = stats["threshold"]

    # If stored threshold is NaN, compute it from the training matrix (95th percentile)
    if threshold is None or (isinstance(threshold, float) and np.isnan(threshold)):
        if os.path.exists(AD_MATRIX_PATH):
            mat = np.load(AD_MATRIX_PATH)
            mean_clean = np.nan_to_num(mean, nan=0.0)
            std_clean = np.where((std == 0) | np.isnan(std), 1.0, std)
            norm = np.nan_to_num((mat - mean_clean) / std_clean, nan=0.0, posinf=0.0, neginf=0.0)
            distances = np.linalg.norm(norm, axis=1)
            threshold = float(np.nanpercentile(distances, 95))
        else:
            threshold = None  # truly unavailable

    return mean, std, threshold

# ---------------------------------------------------------------------------
# Applicability domain check
# ---------------------------------------------------------------------------
def check_ad(feature_vector, mean, std, threshold):
    """Return (within_ad: bool, distance: float) or None if AD unavailable."""
    if threshold is None or (isinstance(threshold, float) and np.isnan(threshold)):
        return None  # AD not available

    mean_clean = np.nan_to_num(mean, nan=0.0)
    std_clean = np.where((std == 0) | np.isnan(std), 1.0, std)
    normalized = np.nan_to_num(
        (feature_vector - mean_clean) / std_clean,
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    distance = float(np.linalg.norm(normalized))
    return distance <= threshold, distance

# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------
def compute_shap_values(model, feature_vector, feature_columns):
    """
    Compute weighted average SHAP values with respect to the Inducer class (index 1).
    Positive SHAP => pushes prediction toward Inducer.
    Negative SHAP => pushes prediction toward Inhibitor.

    model.estimators_[0] = XGBoost (weight=2)
      - shap_values shape: (1, n_features)  [already Inducer log-odds direction]
    model.estimators_[1] = RandomForest (weight=1)
      - shap_values shape: (1, n_features, n_classes)  [axis-2 index 1 = Inducer class]
    """
    x_2d = feature_vector.reshape(1, -1)
    xgb_estimator = model.estimators_[0]
    rf_estimator  = model.estimators_[1]

    # --- XGBoost SHAP ---
    # For binary XGBoost, TreeExplainer returns values for the positive class (Inducer, index 1).
    # Shape is (1, n_features); positive value => increases P(Inducer).
    explainer_xgb = shap.TreeExplainer(xgb_estimator)
    shap_xgb = np.array(explainer_xgb.shap_values(x_2d))
    if shap_xgb.ndim == 3:       # older shap: (n_classes, 1, n_features)
        shap_xgb = shap_xgb[1]  # index 1 = Inducer class
    shap_xgb = shap_xgb.flatten()  # (n_features,)

    # --- Random Forest SHAP ---
    # Recent shap returns (1, n_features, n_classes);
    # axis-2 index 1 corresponds to the Inducer class (label=1).
    explainer_rf = shap.TreeExplainer(rf_estimator)
    shap_rf = np.array(explainer_rf.shap_values(x_2d))
    if shap_rf.ndim == 3:            # (1, n_features, n_classes)
        shap_rf = shap_rf[0, :, 1]  # sample 0, all features, Inducer class
    elif shap_rf.ndim == 2:          # (1, n_features) — already Inducer class
        shap_rf = shap_rf.flatten()
    else:
        shap_rf = shap_rf.flatten()

    # Trim to declared feature count
    n = len(feature_columns)
    shap_xgb = shap_xgb[:n]
    shap_rf   = shap_rf[:n]

    # Weighted average matching ensemble weights (XGB=2, RF=1)
    shap_weighted = (2.0 * shap_xgb + 1.0 * shap_rf) / 3.0
    return shap_weighted

# ---------------------------------------------------------------------------
# SHAP bar chart
# ---------------------------------------------------------------------------
def plot_shap_bar(shap_values, feature_columns, top_n=15):
    abs_shap = np.abs(shap_values)
    top_idx  = np.argsort(abs_shap)[::-1][:top_n]

    top_names = [feature_columns[i] for i in top_idx]
    top_vals  = shap_values[top_idx]

    # Positive SHAP => Inducer-pushing (red); negative => Inhibitor-pushing (blue)
    colors = ["#c0392b" if v > 0 else "#2980b9" for v in top_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_vals[::-1], color=colors[::-1], edgecolor="none", height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (weighted ensemble, Inducer class)", fontsize=10)
    ax.set_title(f"Top {top_n} features by SHAP contribution", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#c0392b", label="Inducer-pushing"),
        Patch(facecolor="#2980b9", label="Inhibitor-pushing"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Functional group detection
# ---------------------------------------------------------------------------
# Corrected functional group definitions derived from paper enrichment analysis.
# Tendencies: Aniline => Inhibitor, Halogen => Inducer,
#             Tertiary amine (non-aniline) => Inducer, Aromatic amine => Inhibitor.
FUNCTIONAL_GROUPS = [
    {
        "name": "Aniline motif",
        "smarts": "c1ccccc1N",
        "tendency": "Inhibitor",
    },
    {
        "name": "Halogen atom",
        "smarts": "[F,Cl,Br,I]",
        "tendency": "Inducer",
    },
    {
        "name": "Tertiary amine",
        "smarts": "[NX3;!$(NC=O);!$(Nc1ccccc1)]",
        "tendency": "Inducer",
    },
    {
        "name": "Aromatic amine",
        "smarts": "[NH2]c1ccccc1",
        "tendency": "Inhibitor",
    },
]

def detect_functional_groups(mol):
    rows = []
    for fg in FUNCTIONAL_GROUPS:
        pattern = Chem.MolFromSmarts(fg["smarts"])
        present = mol.HasSubstructMatch(pattern) if pattern else False
        rows.append({
            "Functional Group": fg["name"],
            "Present": "\u2705" if present else "\u274c",
            "Associated Class Tendency": fg["tendency"],
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Prediction label helper
# ---------------------------------------------------------------------------
def prediction_label(class_idx):
    return "Ferroptosis Inducer" if class_idx == 1 else "Ferroptosis Inhibitor"

def prediction_color(class_idx):
    return "inverse" if class_idx == 1 else "normal"

# ---------------------------------------------------------------------------
# Single molecule prediction
# ---------------------------------------------------------------------------
def run_single_prediction(smiles, model, feature_columns, n_features, ad_mean, ad_std, ad_threshold):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.warning("Invalid SMILES string. Please enter a valid SMILES.")
        return

    # ---- Layout
    col_struct, col_results = st.columns([1, 2], gap="large")

    with col_struct:
        st.subheader("2D Structure")
        img = Draw.MolToImage(mol, size=(300, 250))
        st.image(img, width=300)

        st.subheader("Functional Group Analysis")
        fg_df = detect_functional_groups(mol)
        st.dataframe(fg_df, hide_index=True, width="stretch")

    with col_results:
        # Feature vector
        x = build_feature_vector(smiles, feature_columns, n_features)
        if x is None:
            st.error("Feature vector computation failed.")
            return

        proba = model.predict_proba(x.reshape(1, -1))[0]
        class_idx = int(np.argmax(proba))
        confidence = float(proba[class_idx])
        label = prediction_label(class_idx)

        # Prediction metric
        st.subheader("Prediction")
        st.markdown(
            f"<h2 style='color:{'#c0392b' if class_idx == 1 else '#27ae60'};'>{label}</h2>",
            unsafe_allow_html=True,
        )

        # Class probabilities as percentages
        col_p0, col_p1 = st.columns(2)
        with col_p0:
            st.metric("P(Inhibitor)", f"{proba[0] * 100:.1f}%")
        with col_p1:
            st.metric("P(Inducer)", f"{proba[1] * 100:.1f}%")

        # Applicability domain
        st.subheader("Applicability Domain")
        ad_result = check_ad(x, ad_mean, ad_std, ad_threshold)
        if ad_result is None:
            st.info("Applicability domain check not available (threshold not set).")
        else:
            within_ad, distance = ad_result
            if within_ad:
                st.success(f"Within Applicability Domain  (distance = {distance:.2f})")
            else:
                st.warning(
                    f"Outside Applicability Domain (distance = {distance:.2f}). "
                    "Interpret this prediction with caution."
                )

    # SHAP — full width below
    st.subheader("Feature Importance (SHAP)")
    with st.spinner("Computing SHAP values..."):
        try:
            shap_vals = compute_shap_values(model, x, feature_columns)
            fig = plot_shap_bar(shap_vals, feature_columns, top_n=15)
            st.pyplot(fig, width="content")
            plt.close(fig)
            st.caption(
                "Positive SHAP values push prediction toward Inducer; "
                "negative values push toward Inhibitor."
            )
        except Exception as e:
            st.error(f"SHAP computation failed: {e}")

# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------
def run_batch_prediction(df, model, feature_columns, n_features, ad_mean, ad_std, ad_threshold):
    if "SMILES" not in df.columns:
        st.error("CSV must contain a 'SMILES' column.")
        return

    has_name = "Name" in df.columns
    results = []

    progress_bar = st.progress(0)
    total = len(df)

    for row_num, (i, row) in enumerate(df.iterrows()):
        smiles = str(row["SMILES"]).strip()
        name = str(row["Name"]) if has_name else str(i)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append({
                "Name": name,
                "SMILES": smiles,
                "Prediction": "Invalid SMILES",
                "Probability": None,
                "AD Status": "N/A",
            })
            progress_bar.progress((row_num + 1) / total)
            continue

        x = build_feature_vector(smiles, feature_columns, n_features)
        if x is None:
            results.append({
                "Name": name,
                "SMILES": smiles,
                "Prediction": "Feature Error",
                "Probability": None,
                "AD Status": "N/A",
            })
            progress_bar.progress((row_num + 1) / total)
            continue

        proba = model.predict_proba(x.reshape(1, -1))[0]
        class_idx = int(np.argmax(proba))
        confidence = float(proba[class_idx])
        label = prediction_label(class_idx)

        ad_result = check_ad(x, ad_mean, ad_std, ad_threshold)
        if ad_result is None:
            ad_status = "Unknown"
        else:
            within_ad, _ = ad_result
            ad_status = "Within AD" if within_ad else "Outside AD"

        results.append({
            "Name": name,
            "SMILES": smiles,
            "Prediction": label,
            "Probability": round(confidence, 4),
            "AD Status": ad_status,
        })
        progress_bar.progress((row_num + 1) / total)

    results_df = pd.DataFrame(results)
    progress_bar.empty()

    st.subheader("Prediction Results")
    st.dataframe(results_df, width="stretch")

    # Download
    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results CSV",
        data=csv_bytes,
        file_name="ferroptosis_predictions.csv",
        mime="text/csv",
    )

    # Summary charts
    st.subheader("Summary")
    valid_results = results_df[~results_df["Prediction"].isin(["Invalid SMILES", "Feature Error"])]

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        counts = valid_results["Prediction"].value_counts()
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.bar(
            counts.index,
            counts.values,
            color=["#2980b9" if "Inhibitor" in k else "#c0392b" for k in counts.index],
            edgecolor="none",
        )
        ax1.set_title("Predicted Class Distribution", fontsize=10, fontweight="bold")
        ax1.set_ylabel("Count")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig1, width="stretch")
        plt.close(fig1)

    with col_chart2:
        ad_counts = results_df["AD Status"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ad_colors = {
            "Within AD": "#27ae60",
            "Outside AD": "#e67e22",
            "N/A": "#95a5a6",
            "Unknown": "#95a5a6",
        }
        ax2.bar(
            ad_counts.index,
            ad_counts.values,
            color=[ad_colors.get(k, "#7f8c8d") for k in ad_counts.index],
            edgecolor="none",
        )
        ax2.set_title("Applicability Domain Status", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Count")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2, width="stretch")
        plt.close(fig2)

# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    # Load resources
    model = load_model()
    feature_columns, n_features = load_features()
    ad_mean, ad_std, ad_threshold = load_ad_stats()

    # ---- Sidebar ----
    with st.sidebar:
        st.title("FerroptosisPredictor")
        st.caption("QSAR-based classification of ferroptosis modulators")
        st.divider()

        input_mode = st.radio(
            "Input Mode",
            options=["Single SMILES", "Batch"],
            index=0,
        )

        st.divider()
        with st.expander("About the Model"):
            st.markdown(
                """
                This tool uses a soft-voting ensemble classifier (Random Forest + XGBoost)
                trained on **1624 ferroptosis modulators** to classify compounds as
                Ferroptosis Inducers or Ferroptosis Inhibitors.

                Features are derived from 44 physicochemical molecular descriptors
                and 2048-bit Morgan fingerprints (radius 2).

                Applicability domain is assessed using the normalized Euclidean
                distance from the training set centroid.

                **For research use only.**
                """
            )

    # ---- Main panel ----
    st.title("Ferroptosis Modulator Classification")
    st.markdown(
        "Predict whether a compound acts as a **Ferroptosis Inducer** or "
        "**Ferroptosis Inhibitor** using a trained QSAR ensemble model."
    )
    st.divider()

    if input_mode == "Single SMILES":
        smiles_input = st.text_input(
            "Enter SMILES string",
            placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
        )
        predict_btn = st.button("Predict", type="primary")

        if predict_btn:
            if not smiles_input.strip():
                st.warning("Please enter a SMILES string.")
            else:
                run_single_prediction(
                    smiles_input.strip(),
                    model, feature_columns, n_features,
                    ad_mean, ad_std, ad_threshold,
                )

    elif input_mode == "Batch":
        tab_csv, tab_paste = st.tabs(["CSV Upload", "Paste SMILES"])

        with tab_csv:
            st.markdown(
                "Upload a CSV file with a **SMILES** column (optional: **Name** column). "
                "Each row will be classified independently."
            )
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
                    return

                if st.button("Run Batch Prediction", type="primary", key="btn_csv"):
                    run_batch_prediction(
                        df, model, feature_columns, n_features,
                        ad_mean, ad_std, ad_threshold,
                    )

        with tab_paste:
            st.markdown(
                "Paste one SMILES per line below (optional: add a name separated by a space or tab). "
                "Lines starting with `#` are treated as comments and ignored."
            )
            pasted = st.text_area(
                "SMILES (one per line)",
                height=200,
                placeholder="CC(=O)Oc1ccccc1C(=O)O  Aspirin\nC1=CC=CC=C1  Benzene",
            )
            if st.button("Run Batch Prediction", type="primary", key="btn_paste"):
                if not pasted.strip():
                    st.warning("Please paste at least one SMILES string.")
                else:
                    rows = []
                    for line in pasted.splitlines():
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split(None, 1)  # split on first whitespace
                        smiles = parts[0]
                        name   = parts[1].strip() if len(parts) > 1 else smiles
                        rows.append({"SMILES": smiles, "Name": name})
                    if rows:
                        df_paste = pd.DataFrame(rows)
                        st.write(f"Parsed {len(df_paste)} SMILES.")
                        run_batch_prediction(
                            df_paste, model, feature_columns, n_features,
                            ad_mean, ad_std, ad_threshold,
                        )
                    else:
                        st.warning("No valid SMILES lines found.")

    # ---- Footer ----
    st.divider()
    st.markdown(
        "<div style='text-align:center; color:gray; font-size:0.85rem;'>"
        "Model trained on 1624 ferroptosis modulators | "
        "For research use only | "
        "<a href='https://github.com/Amirtesh/Ferroptosis-Predictor' style='color:gray;'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
