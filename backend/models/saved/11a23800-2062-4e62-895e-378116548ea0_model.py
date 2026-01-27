"""
Trained Gradient Boosting Classifier
Saved on 2026-01-27T12:57:07.301816+00:00
"""
import joblib
from pathlib import Path
from models.prebuilt.gradient_boosting_model import GradientBoostingModel

# Initialize model instance
model = GradientBoostingModel(parameters={'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5})

# Load trained model weights
model_path = Path(__file__).parent / "11a23800-2062-4e62-895e-378116548ea0_model.joblib"
scaler_path = Path(__file__).parent / "11a23800-2062-4e62-895e-378116548ea0_scaler.joblib"

model.model = joblib.load(model_path)
if scaler_path.exists():
    model.scaler = joblib.load(scaler_path)

# Set column configuration
model.TARGET_COLUMN = "Close"
model.FEATURE_COLUMNS = ['DateTime', 'Open', 'High', 'Volume(from bar)', 'Low']
