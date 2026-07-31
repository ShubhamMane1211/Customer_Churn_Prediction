"""Streamlit interface for a single bank-customer churn prediction.

Run after installing the package with ``streamlit run app/app.py``.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit executes this file with ``app/`` as the import root. Adding the
# project's src directory supports direct use without requiring installation.
SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from churn_predictor.artifacts import (
    load_feature_columns,
    load_metadata,
    load_model,
)
from churn_predictor.inference import risk_tier
from churn_predictor.preprocessing import preprocess_inference_data


@st.cache_resource
def load_artifacts():
    """Load the model artifacts once per Streamlit process."""
    return load_model(), load_feature_columns(), load_metadata()


st.set_page_config(page_title="Churn Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Bank Customer Churn Predictor")
st.caption("Enter a customer's details to estimate their likelihood of churning.")

try:
    model, feature_columns, metadata = load_artifacts()
except (FileNotFoundError, ValueError) as error:
    st.error(f"Unable to load model artifacts: {error}")
    st.stop()

threshold = float(metadata["threshold"])

with st.sidebar:
    st.subheader("Model information")
    st.write(f"**Model:** {metadata.get('model_name', 'N/A')}")
    st.write(f"**ROC-AUC:** {metadata.get('roc_auc', 0):.3f}")
    st.write(f"**PR-AUC:** {metadata.get('pr_auc', 0):.3f}")
    st.write(f"**Decision threshold:** {threshold:.3f}")
    st.divider()
    st.markdown(
        f"- 🟢 **Low:** probability < {threshold / 2:.2f}\n"
        f"- 🟡 **Medium:** {threshold / 2:.2f}–{threshold:.2f}\n"
        f"- 🔴 **High:** probability ≥ {threshold:.2f}"
    )

with st.form("customer_form"):
    st.subheader("Customer details")
    left, right = st.columns(2)

    with left:
        credit_score = st.number_input(
            "Credit score", min_value=300, max_value=900, value=650
        )
        country = st.selectbox("Country", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.slider(
            "Tenure (years with bank)", min_value=0, max_value=10, value=5
        )

    with right:
        balance = st.number_input("Balance", min_value=0.0, value=0.0, step=1000.0)
        products_number = st.selectbox("Number of products", [1, 2, 3, 4])
        active_member = st.selectbox("Active member?", ["Yes", "No"])
        credit_card = st.selectbox("Has credit card?", ["Yes", "No"])
        estimated_salary = st.number_input(
            "Estimated salary", min_value=0.0, value=100000.0, step=1000.0
        )

    submitted = st.form_submit_button("Predict churn risk", use_container_width=True)

if submitted:
    customer = pd.DataFrame(
        [
            {
                "credit_score": credit_score,
                "country": country,
                "gender": gender,
                "age": age,
                "tenure": tenure,
                "balance": balance,
                "products_number": products_number,
                "credit_card": int(credit_card == "Yes"),
                "active_member": int(active_member == "Yes"),
                "estimated_salary": estimated_salary,
            }
        ]
    )
    probability = float(
        model.predict_proba(preprocess_inference_data(customer, feature_columns))[:, 1][
            0
        ]
    )
    tier = risk_tier(probability, threshold)

    st.divider()
    first, second = st.columns(2)
    tier_icons = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
    first.metric("Churn probability", f"{probability:.1%}")
    second.metric("Risk tier", f"{tier_icons[tier]} {tier}")
    st.progress(min(probability, 1.0))

    if probability >= threshold:
        st.error(
            f"⚠️ This customer is predicted to **churn** (threshold: {threshold:.2f})."
        )
    else:
        st.success(
            f"✅ This customer is predicted to **stay** (threshold: {threshold:.2f})."
        )
