import os
import pickle
import pandas as pd
import numpy as np
import config

class HotelInferenceEngine:
    """
    Production-grade inference engine to predict booking cancellations in real-time.
    """
    def __init__(self):
        print("Initializing Hotel Inference Engine...")
        
        scaler_path = os.path.join(config.MODELS_DIR, "fitted_scaler.pkl")
        champion_path = os.path.join(config.MODELS_DIR, "champion_model.pkl")
        
        if not os.path.exists(scaler_path) or not os.path.exists(champion_path):
            raise FileNotFoundError(
                "Crucial production artifacts are missing. "
                "Please run preprocessing.py and evaluator.py before deploying inference."
            )
            
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(champion_path, "rb") as f:
            self.model = pickle.load(f)
            
        print("Production artifacts successfully loaded. Engine ready for requests.")

    def _align_and_encode_features(self, raw_input_df):
        """
        Private method to safely transform raw real-time.
        """
        clean_df = raw_input_df.drop(columns=config.LEAKAGE_COLUMNS, errors='ignore')
        
        text_columns = clean_df.select_dtypes(include=['object', 'str']).columns
        encoded_df = pd.get_dummies(clean_df, columns=text_columns, drop_first=True)
        encoded_df.columns = encoded_df.columns.str.replace(' ', '_')
        
        # 3. SCHEMA ALIGNMENT 
        processed_sample_path = os.path.join(config.PROCESSED_DATA_DIR, "X_train.csv")
        training_features = pd.read_csv(processed_sample_path, nrows=1).columns.tolist()
        
        final_encoded_df = encoded_df.reindex(columns=training_features, fill_value=0)
        
        scaler_features = self.scaler.feature_names_in_
        final_encoded_df[scaler_features] = self.scaler.transform(final_encoded_df[scaler_features])
        
        return final_encoded_df

    def predict_single_booking(self, raw_data_dict):
        """
        Evaluates a single customer booking request and extracts structured operational alerts.
        """
        raw_df = pd.DataFrame([raw_data_dict])
        
        processed_input = self._align_and_encode_features(raw_df)
        
        prediction = self.model.predict(processed_input)[0]
        probability = self.model.predict_proba(processed_input)[0][1]
        
        status_verdict = "CANCELLATION RISK" if prediction == 1 else "SAFE BOOKING"
        
        return {
            "verdict": status_verdict,
            "cancellation_probability": f"{probability * 100:.2f}%",
            "raw_score": float(probability)
        }

# ==============================================================================
# LOCAL PRODUCTION SIMULATION (Sanity Check Checkpoint)
# ==============================================================================
if __name__ == "__main__":
    # Initialize the inference pipeline
    engine = HotelInferenceEngine()
    
    # Mocking a complete real-time reservation request arriving from the hotel website API
    mock_new_customer_booking = {
        "arrival_date_year": 2026,          
        "arrival_date_month": "July",
        "arrival_date_week_number": 27,     
        "arrival_date_day_of_month": 15,    
        "lead_time": 340,                   
        "stays_in_weekend_nights": 2,
        "stays_in_week_nights": 5,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Groups",       
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 3,      
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "agent": 1.0,
        "days_in_waiting_list": 0,
        "customer_type": "Transient-Party",
        "adr": 85.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0
    }
    
    print("\n--- Sending New Reservation Request to Inference Engine ---")
    result = engine.predict_single_booking(mock_new_customer_booking)
    
    print("\n==============================================")
    print("HOTEL OPERATIONAL INFERENCE RESPONSE")
    print("==============================================")
    print(f"Booking Status Verdict      : {result['verdict']}")
    print(f"Cancellation Probability    : {result['cancellation_probability']}")
    print("==============================================")