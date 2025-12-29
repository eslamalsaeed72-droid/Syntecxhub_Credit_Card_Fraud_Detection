import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔒",
    layout="centered"
)

# Load model and feature names
@st.cache_resource
def load_model():
    model = joblib.load('xgb_fraud_model.pkl')
    features = joblib.load('feature_names.pkl')
    return model, features

try:
    model, feature_names = load_model()
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'xgb_fraud_model.pkl' and 'feature_names.pkl' are in the same directory.")
    st.stop()

# App title and description
st.title("🔒 Credit Card Fraud Detection System")
st.markdown("""
This app uses an XGBoost model trained on a real-world credit card transaction dataset 
to detect fraudulent transactions in real-time.
""")

st.info("""
**Model Performance Highlights (on unseen test data):**
- ROC-AUC: **0.9775**
- Recall at threshold 0.5: **86.7%** (detects 85 out of 98 fraud cases)
- Best F1-Score: **0.856** at threshold ≈ 0.98
""")

# Sidebar for threshold selection
st.sidebar.header("⚙️ Fraud Detection Settings")
threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.99,
    value=0.50,
    step=0.05,
    help="Lower threshold → Higher fraud detection (more alerts)\nHigher threshold → Fewer false alarms"
)

# Business recommendation based on threshold
if threshold <= 0.40:
    reco = "🟢 **High Sensitivity Mode** – Prioritizing maximum fraud detection (ideal for high-risk environments)"
elif threshold >= 0.70:
    reco = "🔵 **Conservative Mode** – Prioritizing customer experience with minimal false alarms"
else:
    reco = "🟡 **Balanced Mode** – Recommended for most use cases"

st.sidebar.markdown(f"### Current Mode:\n{reco}")

# Input method selection
st.header("🛠 Enter Transaction Details")
input_method = st.radio("Choose input method:", ("Manual Input (Slider)", "Upload CSV File"))

if input_method == "Manual Input (Slider)":
    st.markdown("#### Adjust feature values (V1–V28 are PCA-transformed components)")
    
    # Create input dictionary
    input_data = {}
    cols = st.columns(3)
    
    # Time and Amount first
    with cols[0]:
        input_data['Time'] = st.slider("Time (seconds from first transaction)", 0, 172792, 94813)
    with cols[1]:
        input_data['Amount'] = st.number_input("Transaction Amount ($)", min_value=0.0, value=88.35, step=0.01)
    
    # V1 to V28 sliders (grouped in columns)
    st.markdown("#### PCA Features (V1–V28)")
    for i in range(0, 28, 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < 28:
                feat = f'V{i+j+1}'
                # Default to mean ≈0, range roughly -5 to 5 for most V features
                default_val = 0.0
                min_val = -10.0 if feat in ['V3', 'V10', 'V12', 'V14', 'V17'] else -5.0
                max_val = 5.0
                input_data[feat] = cols[j].slider(feat, min_val, max_val, default_val, step=0.01)

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])[feature_names]

else:  # Upload CSV
    st.markdown("#### Upload a CSV file with one or more transactions")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            if not all(col in input_df.columns for col in feature_names):
                st.error(f"CSV must contain these columns: {', '.join(feature_names)}")
                st.stop()
            input_df = input_df[feature_names]
            st.success(f"Loaded {len(input_df)} transaction(s) successfully!")
            st.dataframe(input_df)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()
    else:
        st.info("Waiting for CSV upload...")
        st.stop()

# Prediction
with st.spinner("Analyzing transaction(s)..."):
    probas = model.predict_proba(input_df)[:, 1]
    predictions = (probas >= threshold).astype(int)

# Results
st.header("🔍 Prediction Results")

for idx, (proba, pred) in enumerate(zip(probas, predictions)):
    if len(input_df) > 1:
        st.subheader(f"Transaction {idx + 1}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Fraud Probability", f"{proba:.1%}")
    
    with col2:
        if pred == 1:
            st.error("🚨 **FRAUD DETECTED**")
        else:
            st.success("✅ **LEGITIMATE**")
    
    with col3:
        if pred == 1:
            st.warning(f"Action: **Block / Review** (Threshold: {threshold})")
        else:
            st.info(f"Action: **Approve** (Threshold: {threshold})")
    
    if len(input_df) > 1:
        st.markdown("---")

# Summary if multiple transactions
if len(input_df) > 1:
    fraud_count = sum(predictions)
    st.markdown(f"""
    ### Batch Summary
    - Total transactions: {len(input_df)}
    - Detected as fraud: **{fraud_count}** ({fraud_count/len(input_df):.1%})
    - Approved: {len(input_df) - fraud_count}
    """)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using XGBoost + Streamlit | Dataset: Kaggle Credit Card Fraud Detection")
