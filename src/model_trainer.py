import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import config

def load_processed_data():
    """
    Loads the synchronized train and test partitions from the processed data directory.
    """
    print("Loading processed dataset partitions...")
    X_train = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "y_test.csv")).squeeze()
    
    print(f"Data partitions loaded successfully. Train size: {X_train.shape} | Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def initialize_models():
    """
    Initializes the models.
    """
    models = {
        "Logistic_Regression": LogisticRegression(
            max_iter=1000, 
            random_state=config.RANDOM_STATE
        ),
        "Decision_Tree": DecisionTreeClassifier(
            max_depth=10,                      
            random_state=config.RANDOM_STATE
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100, 
            random_state=config.RANDOM_STATE,
            n_jobs=-1                         
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=config.RANDOM_STATE,
            eval_metric="logloss",            
            n_jobs=-1
        )
    }
    return models

def run_traditional_training_pipeline():
    print("--- Starting Machine Learning Training Pipeline ---")
    
    # 1. Load the processed data partitions
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # 2. Initialize the dictionary of models
    models = initialize_models()
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # 3. Automated Training and Serialization Loop
    print("\n--- Training Traditional Machine Learning Models ---")
    for name, model in models.items():
        print(f"Training architecture: {name}...")
        
        model.fit(X_train, y_train)
        
        model_filename = f"{name.lower()}_model.pkl"
        model_path = os.path.join(config.MODELS_DIR, model_filename)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        print(f"--> Saved trained artifact at: {model_path}")
        
    print("\n--- Traditional Training Phase Completed Successfully ---")

if __name__ == "__main__":
    run_traditional_training_pipeline()