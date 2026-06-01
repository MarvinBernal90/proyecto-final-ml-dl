import streamlit as st
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

# Ensure the system locates the src directory and the processed artifacts
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.predictor import HotelInferenceEngine
import config

# Web application global layout configuration
st.set_page_config(page_title="Hotel Predictive Analytics", page_icon="🏨", layout="centered")

st.title("🏨 Hotel Booking Cancellation Engine")
st.markdown("---")

# Setup the dual-tab interface navigation system
tab1, tab2 = st.tabs(["🔮 Real-Time Predictor", "📊 Model Analytics (QA)"])

# Initialize the production engine using cache resource to enable instant reload
@st.cache_resource
def get_inference_engine():
    return HotelInferenceEngine()

try:
    engine = get_inference_engine()
except Exception as e:
    st.error(f"Error loading production artifacts: {e}")
    st.stop()

# ==============================================================================
# TAB 1: REAL-TIME PRODUCTION INFERENCE FORM
# ==============================================================================
with tab1:
    st.subheader("Operational Risk Assessment")
    st.write("Enter reservation details below to analyze the cancellation risk via the Champion Model.")
    
    with st.form("prediction_form"):
        st.subheader("Booking Characteristics")
        col1, col2 = st.columns(2)
        with col1:
            lead_time = st.number_input("Lead Time (Days in advance)", min_value=0, max_value=730, value=120)
            market_segment = st.selectbox("Market Segment", ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate"])
            deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refundable", "Refundable"])
        with col2:
            adr = st.number_input("Average Daily Rate (ADR - €)", min_value=0.0, value=95.0)
            customer_type = st.selectbox("Customer Type", ["Transient", "Transient-Party", "Contract", "Group"])
            country = st.text_input("Guest Country Code", value="PRT").upper()
            required_car_parking_spaces = st.selectbox("Required Car Parking Spaces", [0, 1, 2])

        st.markdown("---")
        st.subheader("Guest Historical Behavior")
        col3, col4 = st.columns(2)
        with col3:
            previous_cancellations = st.number_input("Previous Cancellations Count", min_value=0, value=0)
        with col4:
            previous_bookings_not_canceled = st.number_input("Previous Successful Bookings", min_value=0, value=0)
            total_of_special_requests = st.number_input("Total Special Requests", min_value=0, value=1)

        submit_button = st.form_submit_button(label="Analyze Cancellation Risk 🚀")

    if submit_button:
        # Construct the payload dictionary adding standard baseline defaults for unexposed variables
        input_data = {
            "arrival_date_year": 2026, "arrival_date_month": "July", "arrival_date_week_number": 27, "arrival_date_day_of_month": 15,
            "lead_time": lead_time, "stays_in_weekend_nights": 1, "stays_in_week_nights": 3, "adults": 2, "children": 0, "babies": 0,
            "meal": "BB", "country": country, "market_segment": market_segment, "distribution_channel": "TA/TO", "is_repeated_guest": 0,
            "previous_cancellations": previous_cancellations, "previous_bookings_not_canceled": previous_bookings_not_canceled,
            "reserved_room_type": "A", "assigned_room_type": "A", "booking_changes": 0, "deposit_type": deposit_type,
            "agent": 1.0, "days_in_waiting_list": 0, "customer_type": customer_type, "adr": adr,
            "required_car_parking_spaces": required_car_parking_spaces, "total_of_special_requests": total_of_special_requests
        }
        
        # Trigger real-time feature engineering pipeline and architectural model inference
        with st.spinner("Analyzing data patterns via Champion Model..."):
            result = engine.predict_single_booking(input_data)
        
        st.markdown("### Operational Verdict")
        if "RISK" in result["verdict"]:
            st.error(f"### {result['verdict']}")
            st.metric(label="Probability of Cancellation", value=result["cancellation_probability"])
            st.warning("⚠️ Action Required: High-risk profile. Consider requiring a deposit or card guarantee.")
        else:
            st.success(f"### {result['verdict']}")
            st.metric(label="Probability of Cancellation", value=result["cancellation_probability"])
            st.info("💡 Safe booking. No immediate mitigation actions needed.")

# ==============================================================================
# TAB 2: TECHNICAL MODEL AUDITING & SYSTEM QUALITY ASSURANCE (QA)
# ==============================================================================
with tab2:
    st.subheader("📊 Production Model Performance Metrics")
    st.write("Scientific validation of the deployed architecture based on the test set partition.")
    
    # Safely load the processed test data matrices to feed the plotting functions
    try:
        X_test = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "X_test.csv"))
        y_test = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "y_test.csv")).squeeze()
        X_train_sample = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "X_train.csv"), nrows=1)
        
        # Apply standard visualization styles for production-ready reports
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12})

        # --- 1. CONFUSION MATRIX ---
        st.markdown("### 1. Confusion Matrix with Operational Labels")
        y_pred = engine.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Not Canceled (0)", "Canceled (1)"],
                    yticklabels=["Not Canceled (0)", "Canceled (1)"], ax=ax1)
        ax1.set_xlabel("Predicted Booking Status", labelpad=10)
        ax1.set_ylabel("Actual Booking Status", labelpad=10)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

        # --- 2. COMPARATIVE ROC CURVE ---
        st.markdown("### 2. Comparative ROC Curves")
        st.write("*Note: This curve represents the performance of the selected Champion Model.*")
        
        y_proba = engine.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.plot(fpr, tpr, color="green", linewidth=3, label=f"Random Forest (Champion) (AUC = {roc_auc:.4f})")
        ax2.plot([0, 1], [0, 1], color="darkgray", linestyle="--", label="Random Guessing (AUC = 0.5000)")
        ax2.set_xlim([-0.02, 1.02])
        ax2.set_ylim([-0.02, 1.02])
        ax2.set_xlabel("False Positive Rate (1 - Specificity)", labelpad=10)
        ax2.set_ylabel("True Positive Rate (Sensitivity)", labelpad=10)
        ax2.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # --- 3. FEATURE IMPORTANCE ---
        st.markdown("### 3. Feature Importance Mapping (Champion Model)")
        if hasattr(engine.model, "feature_importances_"):
            importances = engine.model.feature_importances_
            feature_names = X_train_sample.columns
            indices = np.argsort(importances)[::-1][:15]
            
            fig3, ax3 = plt.subplots(figsize=(7, 5.5))
            sns.barplot(x=importances[indices], y=feature_names[indices], palette="viridis", 
                        hue=feature_names[indices], legend=False, ax=ax3)
            ax3.set_xlabel("Relative Mathematical Importance Score", labelpad=10)
            ax3.set_ylabel("Feature/Variable Name", labelpad=10)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
        else:
            st.info("The selected model architecture does not support feature_importances_ natively.")

    except Exception as err:
        st.warning(f"Could not load test datasets to generate performance graphs: {err}")