import os

# ==============================================================================
# PROJECT DIRECTORY PATHS
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# File path constants
DATA_FILE_PATH = os.path.join(RAW_DATA_DIR, "hotel_bookings.csv")

# ==============================================================================
# PIPELINE CONFIGURATION PARAMETERS
# ==============================================================================
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Target column definition
TARGET_COLUMN = "is_canceled"

# Features to drop immediately to eliminate data leakage
LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date", "company"]