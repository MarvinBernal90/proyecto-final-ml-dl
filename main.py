import os
import sys
import time

# Ensure the project root and src directory are in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import the main pipeline functions from your modules
from src.preprocessing import run_preprocessing_pipeline
from src.model_trainer import run_traditional_training_pipeline
from src.nn_trainer import run_nn_training_pipeline
from src.evaluator import run_evaluation_pipeline

def run_end_to_end_mlops_pipeline():
    """
    Centralized orchestrator that executes the entire Machine Learning pipeline
    from raw data processing to champion model serialization.
    """
    print("="*80)
    print("🚀 INITIALIZING END-TO-END AUTOMATED MLOPS PIPELINE")
    print("="*80)
    
    start_total_time = time.time()
    
    # --------------------------------------------------------------------------
    # STAGE 1: Data Preprocessing & Feature Engineering
    # --------------------------------------------------------------------------
    print("\n[STAGE 1/4] Executing Data Preprocessing...")
    start_time = time.time()
    run_preprocessing_pipeline()
    print(f"✔️ Stage 1 Completed. Duration: {time.time() - start_time:.2f} seconds.")
    
    # --------------------------------------------------------------------------
    # STAGE 2: Traditional Machine Learning Training
    # --------------------------------------------------------------------------
    print("\n[STAGE 2/4] Executing Traditional ML Model Training...")
    start_time = time.time()
    run_traditional_training_pipeline()
    print(f"✔️ Stage 2 Completed. Duration: {time.time() - start_time:.2f} seconds.")
    
    # --------------------------------------------------------------------------
    # STAGE 3: Deep Learning Neural Network Training
    # --------------------------------------------------------------------------
    print("\n[STAGE 3/4] Executing Deep Learning Architecture Training...")
    start_time = time.time()
    run_nn_training_pipeline()
    print(f"✔️ Stage 3 Completed. Duration: {time.time() - start_time:.2f} seconds.")
    
    # --------------------------------------------------------------------------
    # STAGE 4: Model Performance Auditing & Champion Selection
    # --------------------------------------------------------------------------
    print("\n[STAGE 4/4] Executing Performance Auditing & Model Selection...")
    start_time = time.time()
    run_evaluation_pipeline()
    print(f"✔️ Stage 4 Completed. Duration: {time.time() - start_time:.2f} seconds.")
    
    # --------------------------------------------------------------------------
    # PIPELINE SUMMARY
    # --------------------------------------------------------------------------
    total_duration = time.time() - start_total_time
    print("\n" + "="*80)
    print(f"🎉 MACHINE LEARNING PIPELINE EXECUTED SUCCESSFULLY")
    print(f"Total Execution Time: {total_duration / 60:.2f} minutes ({total_duration:.2f} seconds)")
    print("="*80)

if __name__ == "__main__":
    run_end_to_end_mlops_pipeline()