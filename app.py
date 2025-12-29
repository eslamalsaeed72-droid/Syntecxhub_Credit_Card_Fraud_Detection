import streamlit as st
import pandas as pd
import numpy as np
import os

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔒",
    layout="centered"
)

st.title("🔒 Credit Card Fraud Detection System")
st.markdown("""
This app uses an XGBoost model trained on credit card transactions to detect fraud in real-time.
""")

# ------------------- Safe and Lazy Loading -------------------
model = None
feature_names = None
shap_available = False

# Try to import and load everything safely
try:
    # Lazy import joblib
    import joblib
    
    model_path = os.path.join("models", "xgb_fraud_model.pkl")
    features_path = os.path.join("models", "feature_names.pkl")
    
    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        st.success("✅ Model loaded successfully from /models folder!")
    else:
        raise FileNotFoundError("Model files missing in /models")
        
except ImportError:
    st.warning("⚠️ 'joblib' library not available – model cannot be loaded.")
    st.info("The app is running in demo mode (predictions disabled until fixed).")

except FileNotFoundError:
    st.warning("⚠️ Model files not found in 'models/' folder.")
    st.info("Make sure 'models/xgb_fraud_model.pkl' and 'models/feature_names.pkl' are committed to GitHub.")

except Exception as e:
    st.warning("⚠️ Error loading model.")
    st.code(str(e))

# Try to import shap optionally
try:
    import shap
    import matplotlib.pyplot as plt
    shap_available = True
except ImportError:
    shap_available = False

# If model loaded successfully, continue with full app
if model is not None:
    st.info("""
    **Model Performance:**
    - ROC-AUC: **0.9775**
    - Recall @ 0.5: **86.7%**
    """)

    # Sidebar
    st.sidebar.header("⚙️ Settings")
    threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.99, 0.50, 0.05)

    if threshold <= 0.40:
        mode = "🟢 High Sensitivity – Max fraud detection"
    elif threshold >= 0.70:
        mode = "🔵 Conservative – Min false alarms"
    else:
        mode = "🟡 Balanced – Recommended"

    st.sidebar.markdown(f"### Mode: {mode}")

    # Input
    st.header("🛠 Enter Transaction")
    input_method = st.radio("Input method:", ("Manual Input", "CSV Upload"))

    if input_method == "Manual Input":
        input_data = {}
        cols = st.columns(3)
        with cols[0]:
            input_data['Time'] = st.slider("Time", 0, 172792, 94813)
        with cols[1]:
            input_data['Amount'] = st.number_input("Amount", 0.0, value=88.35)

        st.markdown("#### PCA Features (V1-V28)")
        for i in range(0, 28, 4):
            cols = st.columns(4)
            for j in range(4):
                if i + j < 28:
                    feat = f'V{i+j+1}'
                    input_data[feat] = cols[j].slider(feat, -10.0, 10.0, 0.0, 0.01)

        input_df = pd.DataFrame([input_data])[feature_names]

    else:
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            input_df = pd.read_csv(uploaded)
            input_df = input_df[feature_names]
            st.dataframe(input_df)
        else:
            st.stop()

    # Prediction
    with st.spinner("Predicting..."):
        probas = model.predict_proba(input_df)[:, 1]
        predictions = (probas >= threshold).astype(int)

    # Results
    st.header("🔍 Results")
    for i, (proba, pred) in enumerate(zip(probas, predictions)):
        if len(input_df) > 1:
            st.subheader(f"Transaction {i+1}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fraud Probability", f"{proba:.1%}")
        with col2:
            if pred == 1:
                st.error("🚨 FRAUD DETECTED")
            else:
                st.success("✅ LEGITIMATE")

    # Optional SHAP
    if len(input_df) == 1 and shap_available and st.checkbox("Show SHAP Explanation"):
        with st.spinner("Generating explanation..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            fig, ax = plt.subplots()
            shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0], matplotlib=True, show=False)
            st.pyplot(fig)

else:
    st.info("🔧 App running in limited mode until model is loaded.")
    st.markdown("### Fix steps:")
    st.markdown("- Commit `models/` folder with both .pkl files to GitHub")
    st.markdown("- Reboot the app in Streamlit Cloud")

st.markdown("---")
st.caption("XGBoost Credit Card Fraud Detection | Deployed with Streamlit")
