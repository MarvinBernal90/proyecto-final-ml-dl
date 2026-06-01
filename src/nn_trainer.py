import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import config

def load_processed_data():
    """
    Loads the synchronized train and test partitions from the processed data directory.
    """
    print("Loading processed dataset partitions for the Neural Network...")
    X_train = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(config.PROCESSED_DATA_DIR, "y_test.csv")).squeeze()
    
    return X_train, X_test, y_train, y_test

def build_neural_network(input_dim):
    """
    Compiles a Deep Neural Network (DNN) architecture optimized for tabular classification.
    """
    # Fix random seeds for reproducibility in TensorFlow
    tf.random.set_seed(config.RANDOM_STATE)
    
    model = Sequential([
        # Input Layer & First Hidden Layer
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),  # Prevents overfitting by randomly disabling 30% of neurons per epoch
        
        # Second Hidden Layer
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        # Output Layer (Binary Classification: Canceled or Not)
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the network specifying the loss function and optimization algorithm
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def run_nn_training_pipeline():
    print("--- Starting Deep Learning Training Pipeline ---")
    
    # 1. Load data
    X_train, X_test, y_train, y_test = load_processed_data()
    input_dimension = X_train.shape[1]  
    
    # 2. Build model architecture
    print(f"Building Deep Neural Network. Input features dimension: {input_dimension}")
    nn_model = build_neural_network(input_dim=input_dimension)
    nn_model.summary()  # Prints the structural architecture map in the console
    
    # 3. Define Early Stopping to safeguard against overfitting
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=5,          # Stops if validation loss doesn't improve for 5 consecutive epochs
        restore_best_weights=True
    )
    
    # 4. Train the Neural Network
    print("\n--- Training Deep Neural Network via TensorFlow/Keras ---")
    history = nn_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,           
        batch_size=128,      
        callbacks=[early_stop],
        verbose=1           
    )
    
    # 5. Deep Learning Model Serialization
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    nn_model_path = os.path.join(config.MODELS_DIR, "nn_keras_model.keras")
    
    nn_model.save(nn_model_path)
    print(f"\n--> Deep Learning model successfully trained and saved at: {nn_model_path}")
    print("--- Deep Learning Pipeline Successfully Completed ---")

if __name__ == "__main__":
    run_nn_training_pipeline()