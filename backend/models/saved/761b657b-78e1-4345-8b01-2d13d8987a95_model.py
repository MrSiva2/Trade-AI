"""
Trained Logistic Regression
Saved on 2026-01-27T17:51:29.583734+00:00
"""
import joblib
from pathlib import Path
from models.prebuilt.logistic_regression_model import LogisticRegressionModel

# Initialize model instance
model = LogisticRegressionModel(parameters={'C': 1.0, 'max_iter': 1000})

# Load trained model weights
model_path = Path(__file__).parent / "761b657b-78e1-4345-8b01-2d13d8987a95_model.joblib"
scaler_path = Path(__file__).parent / "761b657b-78e1-4345-8b01-2d13d8987a95_scaler.joblib"

model.model = joblib.load(model_path)
if scaler_path.exists():
    model.scaler = joblib.load(scaler_path)

# Set column configuration
model.TARGET_COLUMN = "Close"
model.FEATURE_COLUMNS = ['DateTime', 'Open', 'High', 'Volume(from bar)', 'Low']
