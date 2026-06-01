import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config

def run_preprocessing_pipeline():

    print("--- Starting Data Preprocessing Pipeline ---")
    
    # 1. Verify and Load the immutable raw dataset
    if not os.path.exists(config.DATA_FILE_PATH):
        raise FileNotFoundError(
            f"Crucial error: Raw data file is missing at the expected path: {config.DATA_FILE_PATH}. "
            f"Please ensure your CSV file is inside data/raw/ directory."
        )

    # Read and display the dataset dimensions    
    df_hotel = pd.read_csv(config.DATA_FILE_PATH)
    print(f"1. Dataset successfully loaded. Initial raw shape: {df_hotel.shape}")

    # 2. Mitigate Data Leakage (Prior to splitting)
    df_hotel = df_hotel.drop(columns=config.LEAKAGE_COLUMNS, errors='ignore')
    print(f"2. Data leakage mitigated. Shape after dropping status columns: {df_hotel.shape}")

    # 3. Handle Missing Values & Anomalies
    if 'adr' in df_hotel.columns:
        df_hotel = df_hotel[df_hotel['adr'] >= 0] 
    if 'children' in df_hotel.columns:
        df_hotel['children'] = df_hotel['children'].fillna(df_hotel['children'].median())
    if 'agent' in df_hotel.columns:
        df_hotel['agent'] = df_hotel['agent'].fillna(df_hotel['agent'].median())
    if 'country' in df_hotel.columns:
        df_hotel['country'] = df_hotel['country'].fillna('Unknown')
        
    # Verification check to ensure 0 nulls remain in the dataset
    remaining_nulls = df_hotel.isnull().sum().sum()
    print(f"3. Missing values imputation completed. Remaining nulls in dataset: {remaining_nulls}")
    print("--> Anomalous negative ADR values removed.")

    # 4. Feature-Target Split (X and y representation)
    X = df_hotel.drop(columns=[config.TARGET_COLUMN])
    y = df_hotel[config.TARGET_COLUMN]
    print(f"4. Feature matrix X shape: {X.shape} | Target vector y shape: {y.shape}")

    # 5. Train-Test Split (80/20 Reproducible Distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=y  # Maintains target class proportions balanced across both splits
    )
    print(f"5. Train-Test Split completed. X_train: {X_train.shape} | X_test: {X_test.shape}")
    
    # 6. Categorical Selection and Independent One-Hot Encoding
    # Detect object/text columns leveraging information exclusively from X_train
    text_columns = X_train.select_dtypes(include=['object', 'str']).columns
    
    # Apply One-Hot Encoding independently to prevent data leakage
    X_train_encoded = pd.get_dummies(X_train, columns=text_columns, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=text_columns, drop_first=True)
    
    # Align feature columns to guarantee identical structure between train and test matrices
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
    
    # Cosmetic cleanup for feature names by replacing spaces with underscores
    X_train_encoded.columns = X_train_encoded.columns.str.replace(' ', '_')
    X_test_encoded.columns = X_test_encoded.columns.str.replace(' ', '_')
    print(f"6. One-Hot Encoding & Feature Alignment completed. New X_train shape: {X_train_encoded.shape}")
    
    # 7. Feature Scaling via StandardScaler
    print("7. Scaling numerical features using StandardScaler...")
    
    # Detect original continuous/numerical columns from the source data
    num_columns = X.select_dtypes(include=['int64', 'float64']).columns
    
    # Initialize the scaling object
    scaler = StandardScaler()
    
    # The scaler computes parameters and transforms the training set exclusively
    X_train_encoded[num_columns] = scaler.fit_transform(X_train_encoded[num_columns])
    
    # The scaler ONLY TRANSFORMS the testing set (preventing learning look-ahead bias)
    X_test_encoded[num_columns] = scaler.transform(X_test_encoded[num_columns])
    
    # Persist the trained scaler object for downstream usage in the Streamlit web application
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    scaler_path = os.path.join(config.MODELS_DIR, "fitted_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    print(f"--> StandardScaler successfully trained on X_train and serialized at: {scaler_path}")

    # 8. Export Processed Partitions into the processed data directory
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    
    # Export fully encoded and scaled feature matrices (*_encoded)
    X_train_encoded.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "X_train.csv"), index=False)
    X_test_encoded.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(config.PROCESSED_DATA_DIR, "y_test.csv"), index=False)
    
    print("--- Preprocessing Pipeline Successfully Completed ---")
    print(f"Final Train features exported: {X_train_encoded.shape} | Final Test features exported: {X_test_encoded.shape}")

if __name__ == "__main__":
    run_preprocessing_pipeline()