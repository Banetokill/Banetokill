import streamlit as st
import joblib
import pandas as pd
import json

# 1. Load trained models
def load_models():
    rf = joblib.load("models/rf_model.joblib")
    lr = joblib.load("models/lr_model.joblib")
    return rf, lr

rf, lr = load_models()

# 2. Load saved metrics
def load_metrics():
    with open("models/metrics.json", "r") as f:
        return json.load(f)

metrics = load_metrics()

# 3. App layout
st.set_page_config(page_title="HEA Phase Predictor", layout="wide")
st.title("🔬 HEA Phase Prediction AI Agent")

# Sidebar: Investor Dashboard
st.sidebar.header("📊 Investor Dashboard")
st.sidebar.subheader("Model Performance")
st.sidebar.markdown(f"- RF Accuracy: **{metrics['random_forest_accuracy']:.2%}**")
st.sidebar.markdown(f"- LR Accuracy: **{metrics['logistic_regression_accuracy']:.2%}**")

st.sidebar.subheader("Market Opportunity")
st.sidebar.markdown(
    "High-entropy alloys are transforming materials engineering. \
"
    "This AI-driven tool enables rapid phase prediction, cutting R&D costs by up to 50% and \
"
    "accelerating time-to-market. \n\n**Projected ROI**: 5x within 3 years."
)

# Main panel: Input & Prediction
st.subheader("🔧 Predict Alloy Phase")
density = st.number_input("Density_calc", value=0.0)
dhmix = st.number_input("ΔHmix", value=0.0)
dsmix = st.number_input("ΔSmix", value=0.0)
dgmix = st.number_input("ΔGmix", value=0.0)
model_choice = st.selectbox("Select Model", ["Random Forest", "Logistic Regression"])

if st.button("Predict Phase"):
    data = pd.DataFrame({
        "Density_calc": [density],
        "dHmix": [dhmix],
        "dSmix": [dsmix],
        "dGmix": [dgmix]
    })
    model = rf if model_choice == "Random Forest" else lr
    prediction = model.predict(data)[0]
    st.success(f"Predicted Phase: **{prediction}**")

# Feature importance chart
st.subheader("🌟 Feature Importance (Random Forest)")
importances = rf.feature_importances_
features = ["Density_calc", "dHmix", "dSmix", "dGmix"]
imp_df = pd.DataFrame({"feature": features, "importance": importances})
st.bar_chart(imp_df.set_index('feature'))
