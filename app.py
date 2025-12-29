import streamlit as st
import pandas as pd
import numpy as np
import os

# Page config - أول حاجة دايمًا
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔒",
    layout="centered"
)

# Title and initial message
st.title("🔒 Credit Card Fraud Detection System")
st.markdown("""
This app uses an XGBoost model to detect fraudulent credit card transactions in real-time.
""")

# ------------------- Safe Model Loading -------------------
@st.cache_resource(show_spinner=False)
def load_model_and_features():
    try:
        import joblib
        
        model_path = os.path.join("models", "xgb_fraud_model.pkl")
        features_path = os.path.join("models", "feature_names.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        
        st.success("✅ Model loaded successfully!")
        return model, feature_names
    
    except ImportError as e:
        st.error("⚠️ Required library missing (joblib). The app will not work until fixed.")
        st.code(f"Import Error: {str(e)}")
        st.stop()
    
    except FileNotFoundError as e:
        st.error("⚠️ Model files not found!")
        st.markdown("""
        **Please check:**
        - Folder `models/` exists in the repository
        - Contains `xgb_fraud_model.pkl` and `feature_names.pkl`
        - Files are committed and pushed to GitHub
        """)
        st.code(str(e))
        st.stop()
    
    except Exception as e:
        st.error("⚠️ Unexpected error while loading the model.")
        st.code(f"Error: {str(e)}")
        st.stop()

# Load model (will stop app gracefully if failed)
with st.spinner("Loading fraud detection model..."):
    model, feature_names = load_model_and_features()

# ------------------- Rest of the app continues normally -------------------
st.info("""
**Model Performance Highlights:**
- ROC-AUC: **0.9775**
- Recall @ 0.5: **86.7%** (detects 85/98 fraud cases)
""")

# Sidebar threshold
st.sidebar.header("⚙️ Detection Settings")
threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.99,
    value=0.50,
    step=0.05,
    help="Lower = More sensitive (more fraud alerts)\nHigher = Fewer false alarms"
)

# Business mode message
if threshold <= 0.40:
    mode = "🟢 **High Sensitivity** – Maximize fraud detection"
elif threshold >= 0.70:
    mode = "🔵 **Conservative** – Prioritize customer experience"
else:
    mode = "🟡 **Balanced** – Recommended for most cases"

st.sidebar.markdown(f"### Mode:\n{mode}")

# ------------------- Input Section (same as before) -------------------
st.header("🛠 Enter Transaction Details")
input_method = st.radio("Choose input method:", ("Manual Input", "Upload CSV File"))

# ... باقي الكود بتاع الـ input والـ prediction زي ما هو ...
# (من غير تغيير، لأن الـ error بيحصل قبل كده أصلاً)
