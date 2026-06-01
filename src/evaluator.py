import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import tensorflow as tf
import config

def load_evaluation_data():
    """
    Loads only the test dataset partitions required for performance auditing.
    """
    print("Loading processed test partitions for performance auditing...")
    X_test = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "y_test.csv")).squeeze()
    return X_test, y_test

def run_evaluation_pipeline():
    print("--- Starting Performance Auditing & Model Selection Pipeline ---")
    
    # 1. Load the test data
    X_test, y_test = load_evaluation_data()
    
    performance_registry = {}
    loaded_models = {}
    
    traditional_models = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
    
    # 2. Load and Evaluate Traditional ML Models (.pkl)
    print("\nAuditing Traditional Machine Learning Models...")
    for name in traditional_models:
        model_path = os.path.join(config.MODELS_DIR, f"{name}_model.pkl")
        
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            
            performance_registry[name] = {"Accuracy": acc, "ROC-AUC": auc, "Type": "Sklearn/XGB"}
            loaded_models[name] = model
            print(f"-> [{name.upper()}] Audited successfully. ROC-AUC: {auc:.4f}")
        else:
            print(f"Warning: Missing artifact for {name} at {model_path}")

    # 3. Load and Evaluate the Deep Learning Neural Network (.keras)
    nn_path = os.path.join(config.MODELS_DIR, "nn_keras_model.keras")
    print("\nAuditing Deep Learning Neural Network...")
    
    if os.path.exists(nn_path):
        nn_model = tf.keras.models.load_model(nn_path)
        
        y_proba_nn = nn_model.predict(X_test, verbose=0).flatten()
        y_pred_nn = (y_proba_nn >= 0.5).astype(int)  # Standard threshold classification
        
        acc_nn = accuracy_score(y_test, y_pred_nn)
        auc_nn = roc_auc_score(y_test, y_proba_nn)
        
        performance_registry["neural_network"] = {"Accuracy": acc_nn, "ROC-AUC": auc_nn, "Type": "TensorFlow/Keras"}
        loaded_models["neural_network"] = nn_model
        print(f"-> [NEURAL_NETWORK] Audited successfully. ROC-AUC: {auc_nn:.4f}")
    else:
        print(f"Warning: Missing artifact for Deep Learning model at {nn_path}")

    # 4. Compare Competitors and Select the Absolute Champion
    print("\n" + "="*65)
    print(f"{'MODEL ARCHITECTURE':<22} | {'TYPE':<16} | {'ACCURACY':<8} | {'ROC-AUC':<8}")
    print("="*65)
    
    best_auc = 0.0
    champion_name = None
    
    for model_name, metrics in performance_registry.items():
        print(f"- {model_name:<20} | {metrics['Type']:<16} | {metrics['Accuracy']:.4f}   | {metrics['ROC-AUC']:.4f}")
        
        if metrics["ROC-AUC"] > best_auc:
            best_auc = metrics["ROC-AUC"]
            champion_name = model_name

    print("="*65)
    print(f"CHAMPION IDENTIFIED: {champion_name.upper()} (ROC-AUC: {best_auc:.4f})")
    
    # 5. Serialize the Absolute Champion
    champion_object = loaded_models[champion_name]
    
    if performance_registry[champion_name]["Type"] == "TensorFlow/Keras":
        # If Keras wins, save it in native format
        champion_path = os.path.join(config.MODELS_DIR, "champion_model.keras")
        champion_object.save(champion_path)
    else:
        # If a traditional model wins, save it via pickle
        champion_path = os.path.join(config.MODELS_DIR, "champion_model.pkl")
        with open(champion_path, "wb") as f:
            pickle.dump(champion_object, f)
            
    print(f"--> Absolute champion artifact successfully serialized at: {champion_path}")
    print("--- Evaluation and Selection Pipeline Completed ---")

if __name__ == "__main__":
    run_evaluation_pipeline()