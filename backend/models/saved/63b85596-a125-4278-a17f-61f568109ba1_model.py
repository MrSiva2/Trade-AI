"""
Trained Random Forest Classifier
Saved on 2026-01-26T19:06:52.034581+00:00
"""
import joblib
from pathlib import Path
from models.prebuilt.random_forest_model import RandomForestModel

# Initialize model instance
model = RandomForestModel(parameters={'n_estimators': 100, 'max_depth': 10, 'random_state': 42})

# Load trained model weights
model_path = Path(__file__).parent / "63b85596-a125-4278-a17f-61f568109ba1_model.joblib"
scaler_path = Path(__file__).parent / "63b85596-a125-4278-a17f-61f568109ba1_scaler.joblib"

model.model = joblib.load(model_path)
if scaler_path.exists():
    model.scaler = joblib.load(scaler_path)

# Set column configuration
model.TARGET_COLUMN = "Close"
model.FEATURE_COLUMNS = ['DateTime', 'Open', 'High', 'Volume(from bar)', 'Low']
