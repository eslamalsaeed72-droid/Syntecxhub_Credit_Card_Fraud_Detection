import streamlit as st
import pandas as pd
import numpy as np
import os

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔒",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🔒 Credit Card Fraud Detection System")
st.markdown("Real-time fraud detection using XGBoost on credit card transactions.")

# --- Safe Model Loading ---
model = None
feature_names = None
shap_available = False

try:
    import joblib

    model_path = "models/xgb_fraud_model.pkl"
    features_path = "models/feature_names.pkl"

    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        st.success("✅ Model loaded successfully!")
    else:
        st.error("❌ Model files not found in 'models/' folder.")

except ImportError as e:
    st.error("❌ Required library missing: joblib")
    st.info("Add 'joblib' to requirements.txt and reboot the app.")

except Exception as e:
    st.error("❌ Error loading model")
    st.exception(e)

# Try to import SHAP (optional)
try:
    import shap
    import matplotlib.pyplot as plt
    shap_available = True
except ImportError:
    shap_available = False

# --- App runs only if model is loaded ---
if model is None:
    st.stop()

# Model info
st.info("**Model Performance** — ROC-AUC: 0.9775 | Recall @ 0.5: 86.7% (85/98 frauds detected)")

# Sidebar - Threshold
st.sidebar.header("⚙️ Detection Settings")
threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.99,
    value=0.50,
    step=0.05,
    help="Lower → More sensitive | Higher → Fewer false alarms"
)

if threshold <= 0.4:
    mode = "🟢 **High Sensitivity** – Maximize fraud detection"
elif threshold >= 0.7:
    mode = "🔵 **Conservative** – Minimize false alarms"
else:
    mode = "🟡 **Balanced** – Recommended"

st.sidebar.markdown(f"### Current Mode\n{mode}")

# --- Input ---
st.header("🛠 Enter Transaction Data")
input_method = st.radio("Input method", ["Manual Input (Sliders)", "Upload CSV File"])

if input_method == "Manual Input (Sliders)":
    st.caption("Adjust Time, Amount, and PCA features (V1-V28)")

    data = {}
    col1, col2 = st.columns(2)
    with col1:
        data["Time"] = st.number_input("Time (seconds)", 0, 172792, 94813)
    with col2:
        data["Amount"] = st.number_input("Amount ($)", 0.0, value=88.35, step=0.01)

    st.markdown("#### PCA Features (V1 to V28)")
    for i in range(0, 28, 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < 28:
                feat = f"V{i+j+1}"
                data[feat] = cols[j].slider(feat, -10.0, 10.0, 0.0, 0.01)

    input_df = pd.DataFrame([data])[feature_names]

else:
    uploaded = st.file_uploader("Upload CSV with transaction(s)", type="csv")
    if uploaded:
        try:
            input_df = pd.read_csv(uploaded)
            input_df = input_df[feature_names]
            st.success(f"Loaded {len(input_df)} transaction(s)")
            st.dataframe(input_df)
        except Exception as e:
            st.error("Error reading CSV. Check column names.")
            st.stop()
    else:
        st.info("Waiting for CSV upload...")
        st.stop()

# --- Prediction ---
with st.spinner("Analyzing transaction(s)..."):
    probas = model.predict_proba(input_df)[:, 1]
    predictions = (probas >= threshold).astype(int)

# --- Results ---
st.header("🔍 Prediction Results")

for i in range(len(input_df)):
    if len(input_df) > 1:
        st.subheader(f"Transaction {i+1}")

    col1, col2, col3 = st.columns(3)
    proba = probas[i]
    pred = predictions[i]

    with col1:
        st.metric("Fraud Probability", f"{proba:.1%}")

    with col2:
        if pred == 1:
            st.error("🚨 FRAUD DETECTED")
        else:
            st.success("✅ LEGITIMATE")

    with col3:
        if pred == 1:
            st.warning(f"Action: Block/Review (threshold: {threshold})")
        else:
            st.info(f"Action: Approve (threshold: {threshold})")

    if len(input_df) > 1:
        st.markdown("---")

# --- Optional SHAP Explanation (single transaction only) ---
if len(input_df) == 1 and shap_available and st.checkbox("Show SHAP Explanation (why this prediction?)"):
    with st.spinner("Generating explanation..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        fig, ax = plt.subplots()
        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_df.iloc[0],
            matplotlib=True,
            show=False
        )
        st.pyplot(fig)
        plt.close()

        st.caption("Red → pushes toward fraud | Blue → pushes toward legitimate")

# Footer
st.markdown("---")
st.caption("Built with XGBoost + Streamlit | Dataset: Kaggle Credit Card Fraud")
