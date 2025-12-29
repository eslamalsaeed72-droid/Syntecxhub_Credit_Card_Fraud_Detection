# 🔒 Credit Card Fraud Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)  
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)  
![XGBoost](https://img.shields.io/badge/XGBoost-State%20of%20the%20Art-orange)  
![SHAP](https://img.shields.io/badge/Interpretability-SHAP-important)  
![License](https://img.shields.io/badge/License-MIT-green)

A real-time **Credit Card Fraud Detection** system built using **XGBoost** and deployed as an interactive web application with **Streamlit**.

This project demonstrates end-to-end machine learning workflow: data exploration, handling severe class imbalance, model training, performance evaluation, business-oriented threshold analysis, **model interpretability using SHAP**, and deployment.

Live Demo: https://your-app-name.streamlit.app

## 🚀 Features

- **Highly Accurate Fraud Detection** using XGBoost (ROC-AUC: **0.9775** on test set)
- Interactive **Streamlit web app** for real-time predictions
- Adjustable **decision threshold** with clear business mode recommendations
- Support for **single transaction input** (sliders) or **batch prediction** via CSV upload
- Detailed **business insights** on precision-recall trade-offs
- **Model Interpretability** with SHAP values to explain individual and global predictions
- Clean, modular, and production-ready code structure

## 📊 Model Performance (Unseen Test Set)

| Metric                  | Value    | Notes                                      |
|-------------------------|----------|--------------------------------------------|
| ROC-AUC                 | 0.9775   | Excellent discrimination ability          |
| Recall @ threshold 0.5  | 86.7%    | Detects 85 out of 98 fraud cases           |
| Precision @ threshold 0.5 | 52.5%  | Reasonable false positive rate             |
| Best F1-Score           | 0.856    | Achieved at threshold ≈ 0.98               |

### Business Threshold Recommendations

- **High Sensitivity (Max Fraud Detection)** → Threshold **0.40** → 89.8% Recall (only 10 missed frauds)
- **Customer-Friendly (Minimize False Alarms)** → Threshold **0.70** → Only 48 false alarms
- **Balanced (Recommended)** → Threshold **0.50** → Solid protection with acceptable customer impact

## 🔍 Model Interpretability with SHAP

To ensure transparency and trust — especially important in financial applications — the model is explained using **SHAP (SHapley Additive exPlanations)**.

### Key Global Insights from SHAP Summary Plot:
- **V14, V17, V12, V10, V16** are consistently the most important features driving fraud predictions (negative SHAP values indicate higher fraud risk).
- **V11 and V4** also show strong positive impact on fraud probability.
- Lower values in protective features (e.g., V17, V14) significantly increase the predicted risk of fraud.

These findings align perfectly with domain knowledge and correlation analysis during EDA.

> In a future version, per-transaction SHAP force plots will be integrated directly into the Streamlit app for real-time individual prediction explanations.

## 🗂 Project Structure

```
credit-card-fraud-detection/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── models/
│   ├── xgb_fraud_model.pkl    # Trained XGBoost model (best performer)
│   └── feature_names.pkl      # Feature column order
├──  Syntecxhub_Credit_Card_Fraud_Detection.ipynb
└── README.md                  # Project documentation
├── Demo/
│   └── Screenshots to Project
```

## 🛠 Installation & Local Run

1. Clone the repository
```bash
git clone https://github.com/eslamalsaeed72-droid/Syntecxhub_Credit_Card_Fraud_Detection.git
cd credit-card-fraud-detection
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app locally
```bash
streamlit run app.py
```


## ☁️ Deployment on Streamlit Community Cloud

This app is designed for seamless deployment on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push your code to GitHub (this repo)
2. Go to https://share.streamlit.io
3. Click "New app"
4. Connect your GitHub repo → Select branch → Main file path: `app.py`
5. Click "Deploy"

Make sure `models/` folder and both `.pkl` files are committed to the repo.

## 📈 Methodology Overview

- **Dataset**: Kaggle - Credit Card Fraud Detection (284,807 transactions, highly imbalanced ~0.17% fraud)
- **Preprocessing**: Stratified train-test split
- **Imbalance Handling**: SMOTE oversampling on training data
- **Models Compared**: Random Forest vs XGBoost
- **Winner**: XGBoost (superior ROC-AUC and recall)
- **Evaluation**: Precision, Recall, F1, ROC-AUC, Confusion Matrix, Precision-Recall Curve
- **Interpretability**: SHAP values for global and (planned) local explanations
- **Business Focus**: Threshold tuning based on real-world trade-offs

## 📌 Future Improvements

- Integrate **per-transaction SHAP force plots** in the Streamlit app
- Add real-time monitoring dashboard
- Experiment with neural networks or ensemble methods
- Add anomaly detection fallback model

## 👨‍💻 Author

- **Your Name**  
- LinkedIn:https://www.linkedin.com/in/eslam-alsaeed-1a23921aa 
- GitHub: https://github.com/eslamalsaeed72-droid

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ If you found this project helpful, please give it a star!  
💬 Feel free to open issues or submit pull requests.




