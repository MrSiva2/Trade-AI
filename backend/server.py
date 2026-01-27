from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import StreamingResponse, FileResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import json
import asyncio
import aiofiles
import joblib
import importlib.util
import sys
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import threading
import queue
import io

ROOT_DIR = Path(__file__).parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
    
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', "mongodb://localhost:27017")
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'test_database')]

app = FastAPI()
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state for training sessions and logs
training_sessions: Dict[str, Dict] = {}
model_instances: Dict[str, Any] = {}
log_queues: Dict[str, queue.Queue] = {}
output_queues: Dict[str, queue.Queue] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}

# Default data folder - use local project relative paths if /app doesn't exist or we are on Windows
def get_safe_path(env_var, default_rel):
    val = os.environ.get(env_var)
    if not val or (val.startswith('/app') and os.name == 'nt'):
        return str(ROOT_DIR / default_rel)
    return val

DATA_FOLDER = get_safe_path('DATA_FOLDER', 'data')
MODEL_FOLDER = get_safe_path('MODEL_FOLDER', 'models')
PREBUILT_MODELS_FOLDER = ROOT_DIR / 'models' / 'prebuilt'
SAVED_MODELS_FOLDER = Path(MODEL_FOLDER) / 'saved'

# Fallback for previous Docker-style paths on Windows
FALLBACK_SAVED_MODELS_FOLDER = Path("C:/app/models/saved") if os.name == 'nt' else None
FALLBACK_MODEL_FOLDER = Path("C:/app/models") if os.name == 'nt' else None

# Ensure folders exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(SAVED_MODELS_FOLDER, exist_ok=True)
os.makedirs(PREBUILT_MODELS_FOLDER, exist_ok=True)

# Model class cache
_model_class_cache: Dict[str, Any] = {}

# Pydantic Models
class FolderPath(BaseModel):
    path: str

class CSVFileInfo(BaseModel):
    name: str
    path: str
    size: int
    rows: Optional[int] = None
    columns: Optional[List[str]] = None

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: str  # random_forest, gradient_boosting, logistic_regression, lstm, custom
    parameters: Dict[str, Any] = {}
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class CustomModelConfig(BaseModel):
    name: str
    layers: List[Dict[str, Any]]
    optimizer: str = "adam"
    loss: str = "binary_crossentropy"
    learning_rate: float = 0.001

class TrainingRequest(BaseModel):
    model_id: str
    model_name: Optional[str] = None
    train_data_path: str
    test_data_path: Optional[str] = None
    target_column: str
    feature_columns: List[str]
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    nth_candle: Optional[int] = None  # For validation predictions

class SaveModelRequest(BaseModel):
    name: Optional[str] = None

class BacktestRequest(BaseModel):
    model_id: str
    test_data_path: str
    initial_capital: float = 10000.0
    position_size: float = 0.1
    target_candle: int = 1  # Which candle ahead to target (1st, 2nd, 3rd, etc.)
    rr_ratio: Optional[float] = None  # Optional Risk-to-Reward Ratio
    commission_fee: float = 0.0  # Fixed fee per trade
    # target_column and feature_columns will come from the model file

class TrainingSession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    model_name: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed, stopped
    current_epoch: int = 0
    total_epochs: int = 0
    metrics: Dict[str, List[float]] = {}
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

class BacktestResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    trades: List[Dict[str, Any]] = []
    price_data: List[Dict[str, Any]] = []
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# Load pre-built models from .py files
def load_prebuilt_models():
    """Load pre-built model metadata from .py files"""
    models = []
    model_file_map = {
        "rf_default": "random_forest_model.py",
        "gb_default": "gradient_boosting_model.py",
        "lr_default": "logistic_regression_model.py",
        "lstm_default": "lstm_model.py"
    }
    
    for model_id, filename in model_file_map.items():
        model_path = PREBUILT_MODELS_FOLDER / filename
        if model_path.exists():
            try:
                # Load the model class to get metadata
                spec = importlib.util.spec_from_file_location(f"model_{model_id}", model_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[f"model_{model_id}"] = module
                spec.loader.exec_module(module)
                
                # Find the model class (it will be the class with MODEL_TYPE attribute)
                model_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        hasattr(attr, 'MODEL_TYPE') and 
                        hasattr(attr, 'MODEL_ID') and
                        attr.MODEL_ID == model_id):
                        model_class = attr
                        break
                
                if model_class:
                    # Create instance to get default parameters
                    instance = model_class()
                    models.append({
                        "id": model_class.MODEL_ID,
                        "name": model_class.MODEL_NAME,
                        "type": model_class.MODEL_TYPE,
                        "parameters": instance.parameters,
                        "description": model_class.DESCRIPTION,
                        "file_path": str(model_path)
                    })
            except Exception as e:
                logger.error(f"Failed to load model {model_id} from {filename}: {e}")
    
    return models

def get_prebuilt_models():
    """Get cached or load pre-built models"""
    if not hasattr(get_prebuilt_models, '_cache'):
        get_prebuilt_models._cache = load_prebuilt_models()
    return get_prebuilt_models._cache

def load_model_class(model_id: str):
    """Load a model class from .py file"""
    if model_id in _model_class_cache:
        return _model_class_cache[model_id]
    
    model_file_map = {
        "rf_default": ("random_forest_model.py", "RandomForestModel"),
        "gb_default": ("gradient_boosting_model.py", "GradientBoostingModel"),
        "lr_default": ("logistic_regression_model.py", "LogisticRegressionModel"),
        "lstm_default": ("lstm_model.py", "LSTMModel")
    }
    
    if model_id not in model_file_map:
        raise ValueError(f"Unknown model ID: {model_id}")
    
    filename, class_name = model_file_map[model_id]
    model_path = PREBUILT_MODELS_FOLDER / filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        spec = importlib.util.spec_from_file_location(f"model_{model_id}", model_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"model_{model_id}"] = module
        spec.loader.exec_module(module)
        
        model_class = getattr(module, class_name)
        _model_class_cache[model_id] = model_class
        return model_class
    except Exception as e:
        logger.error(f"Failed to load model class {class_name} from {filename}: {e}")
        raise

def add_log(session_id: str, message: str, level: str = "INFO"):
    """Add log message to session queue"""
    if session_id not in log_queues:
        log_queues[session_id] = queue.Queue()
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message
    }
    log_queues[session_id].put(log_entry)
    logger.info(f"[{session_id}] {message}")

def add_output(session_id: str, output: Dict[str, Any]):
    """Add model output to session queue"""
    if session_id not in output_queues:
        output_queues[session_id] = queue.Queue()
    output_queues[session_id].put(output)

def create_model(model_config: Dict[str, Any]):
    """Create sklearn model from config"""
    model_type = model_config.get("type", "random_forest")
    params = model_config.get("parameters", {})
    
    if model_type == "random_forest":
        return RandomForestClassifier(**params)
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(**params)
    elif model_type == "logistic_regression":
        return LogisticRegression(**params)
    else:
        return RandomForestClassifier(n_estimators=100, random_state=42)

async def run_training(session_id: str, request: TrainingRequest, model_instance):
    """Background training task using model class methods"""
    try:
        training_sessions[session_id]["status"] = "running"
        training_sessions[session_id]["started_at"] = datetime.now(timezone.utc).isoformat()
        
        add_log(session_id, f"Starting training for model: {model_instance.MODEL_NAME}")
        add_log(session_id, f"Loading data from: {request.train_data_path}")
        
        # Load data
        df = pd.read_csv(request.train_data_path)
        add_log(session_id, f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Use model's preprocess_data method
        try:
            X, y, scaler = model_instance.preprocess_data(df, request.target_column, request.feature_columns)
            add_log(session_id, f"Data preprocessed: {X.shape[1]} features, {len(y)} samples")
            add_log(session_id, f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        except Exception as e:
            raise ValueError(f"Data preprocessing failed: {str(e)}")
        
        # Train using model's train method
        total_epochs = request.epochs
        training_sessions[session_id]["total_epochs"] = total_epochs
        
        add_log(session_id, f"Starting training with {total_epochs} epochs, validation split: {request.validation_split}")
        if request.nth_candle:
            add_log(session_id, f"Using nth candle validation: {request.nth_candle}")
        
        # Train the model (this will simulate epochs internally)
        trained_model, metrics = model_instance.train(
            X, y, 
            epochs=total_epochs,
            validation_split=request.validation_split,
            nth_candle=request.nth_candle,
            scaler=scaler
        )
        
        # Update metrics in real-time (simulate progress)
        for epoch in range(total_epochs):
            if training_sessions[session_id]["status"] == "stopped":
                add_log(session_id, "Training stopped by user", "WARNING")
                break
            
            training_sessions[session_id]["current_epoch"] = epoch + 1
            training_sessions[session_id]["metrics"] = {
                "loss": metrics["loss"][:epoch+1],
                "accuracy": metrics["accuracy"][:epoch+1],
                "val_loss": metrics["val_loss"][:epoch+1],
                "val_accuracy": metrics["val_accuracy"][:epoch+1]
            }
            
            # Add output with predictions every 10 epochs
            if epoch % 10 == 0 and epoch < len(metrics["val_accuracy"]):
                # Get validation predictions for display
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=request.validation_split, random_state=42
                )
                val_pred, val_probs = model_instance.test(X_val[:10])
                
                add_output(session_id, {
                    "epoch": epoch + 1,
                    "predictions": val_pred[:10].tolist() if hasattr(val_pred, 'tolist') else list(val_pred[:10]),
                    "actual": y_val[:10].tolist() if hasattr(y_val, 'tolist') else list(y_val[:10]),
                    "probabilities": val_probs[:10].tolist() if hasattr(val_probs, 'tolist') else list(val_probs[:10])
                })
            
            loss = metrics["loss"][epoch] if epoch < len(metrics["loss"]) else 0
            acc = metrics["accuracy"][epoch] if epoch < len(metrics["accuracy"]) else 0
            val_loss = metrics["val_loss"][epoch] if epoch < len(metrics["val_loss"]) else 0
            val_acc = metrics["val_accuracy"][epoch] if epoch < len(metrics["val_accuracy"]) else 0
            
            add_log(session_id, f"Epoch {epoch + 1}/{total_epochs} - loss: {loss:.4f} - accuracy: {acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
            
            await asyncio.sleep(0.1)  # Small delay for real-time updates
        
        # Store model instance in separate global dict for later saving (not in training_sessions)
        model_instances[session_id] = model_instance
        # Don't overwrite model_name if request.model_name is empty
        if request.model_name:
            training_sessions[session_id]["model_name"] = request.model_name
            
        training_sessions[session_id]["target_column"] = request.target_column
        training_sessions[session_id]["feature_columns"] = request.feature_columns
        
        training_sessions[session_id]["status"] = "completed"
        training_sessions[session_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        add_log(session_id, "Training completed successfully! Click 'Save Model' to save the trained model.", "SUCCESS")
        
        # Store in MongoDB (handle connection errors)
        try:
            await db.training_sessions.update_one(
                {"id": session_id},
                {"$set": training_sessions[session_id]},
                upsert=True
            )
        except Exception as db_error:
            logger.error(f"Failed to update MongoDB: {db_error}")
            add_log(session_id, "Note: Training completed but failed to save status to permanent database. Model is still available to save.", "WARNING")
        
    except Exception as e:
        if session_id in training_sessions:
            training_sessions[session_id]["status"] = "failed"
        add_log(session_id, f"Training failed: {str(e)}", "ERROR")
        logger.error(f"Training error: {e}")

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Trading AI Model Hub API", "version": "1.0.0"}

@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

# Data Management Routes
@api_router.get("/data/folders")
async def list_folders():
    """List available data folders"""
    folders = []
    base_path = Path(DATA_FOLDER)
    
    if base_path.exists():
        folders.append({"path": str(base_path), "name": base_path.name})
        for folder in base_path.iterdir():
            if folder.is_dir():
                folders.append({"path": str(folder), "name": folder.name})
    
    return {"folders": folders, "base_path": DATA_FOLDER}

@api_router.post("/data/set-folder")
async def set_data_folder(folder: FolderPath):
    """Set the data folder path"""
    path = Path(folder.path)
    if not path.exists():
        os.makedirs(path, exist_ok=True)
    return {"success": True, "path": str(path)}

@api_router.get("/data/files")
async def list_csv_files(folder: str = None, lightweight: bool = Query(default=False)):
    """List CSV files in folder
    
    Args:
        folder: Optional folder path to list files from
        lightweight: If True, skip row counting and column reading for faster loading
    """
    folder_path = Path(folder) if folder else Path(DATA_FOLDER)
    files = []
    
    if folder_path.exists():
        # Use os.scandir for faster directory listing (faster than glob)
        import os
        try:
            with os.scandir(folder_path) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith('.csv'):
                        try:
                            # Use entry.stat() which is faster than Path.stat()
                            stat_info = entry.stat()
                            file_info = {
                                "name": entry.name,
                                "path": str(entry.path),
                                "size": stat_info.st_size
                            }
                            
                            if not lightweight:
                                # Only read columns and count rows if not in lightweight mode
                                try:
                                    # Read only header row for columns (much faster)
                                    df_header = pd.read_csv(entry.path, nrows=0)
                                    file_info["columns"] = list(df_header.columns)
                                    
                                    # Count rows (this is the slowest operation - only do when needed)
                                    file_info["rows"] = sum(1 for _ in open(entry.path)) - 1
                                except Exception:
                                    pass  # Keep basic info even if reading fails
                            
                            files.append(file_info)
                        except Exception:
                            # If stat() fails, still try to add basic info
                            files.append({
                                "name": entry.name,
                                "path": str(entry.path),
                                "size": 0
                            })
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            # Fallback to glob if scandir fails
            for file in folder_path.glob("*.csv"):
                try:
                    stat_info = file.stat()
                    file_info = {
                        "name": file.name,
                        "path": str(file),
                        "size": stat_info.st_size
                    }
                    if not lightweight:
                        try:
                            df_header = pd.read_csv(file, nrows=0)
                            file_info["columns"] = list(df_header.columns)
                            file_info["rows"] = sum(1 for _ in open(file)) - 1
                        except Exception:
                            pass
                    files.append(file_info)
                except Exception:
                    files.append({
                        "name": file.name,
                        "path": str(file),
                        "size": 0
                    })
    
    return {"files": files}

@api_router.delete("/data/files")
async def delete_csv_file(file_path: str):
    """Delete a CSV file"""
    try:
        path = Path(file_path)
        if path.exists() and path.is_file() and path.suffix.lower() == '.csv':
            os.remove(path)
            return {"success": True}
        else:
            raise HTTPException(status_code=404, detail="File not found or not a CSV")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.delete("/data/folders")
async def delete_folder(folder_path: str):
    """Delete a folder and its contents"""
    try:
        path = Path(folder_path)
        if path.exists() and path.is_dir():
            import shutil
            shutil.rmtree(path)
            return {"success": True}
        else:
            raise HTTPException(status_code=404, detail="Folder not found")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@api_router.get("/data/preview")
async def preview_csv(file_path: str, rows: int = 100):
    """Preview CSV file content"""
    try:
        df = pd.read_csv(file_path, nrows=rows)
        return {
            "columns": list(df.columns),
            "data": df.to_dict(orient="records"),
            "total_rows": sum(1 for _ in open(file_path)) - 1,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/data/upload")
async def upload_csv(files: List[UploadFile] = File(...)):
    """Upload one or more CSV files"""
    uploaded_files = []
    failed_files = []
    
    try:
        for file in files:
            try:
                if not file.filename.lower().endswith('.csv'):
                    failed_files.append({"filename": file.filename, "reason": "Not a CSV file"})
                    continue
                    
                file_path = os.path.join(DATA_FOLDER, file.filename)
                async with aiofiles.open(file_path, 'wb') as f:
                    content = await file.read()
                    await f.write(content)
                
                uploaded_files.append({
                    "filename": file.filename,
                    "path": file_path,
                    "size": len(content)
                })
            except Exception as e:
                failed_files.append({"filename": file.filename, "reason": str(e)})
        
        return {
            "success": True, 
            "uploaded": uploaded_files,
            "failed": failed_files,
            "count": len(uploaded_files)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Model Management Routes
@api_router.get("/models/prebuilt")
async def get_prebuilt_models_endpoint():
    """Get list of pre-built models"""
    models = get_prebuilt_models()
    return {"models": models}

@api_router.get("/models/prebuilt/download/{model_id}")
async def download_prebuilt_model(model_id: str):
    """Download a pre-built model .py file"""
    model_file_map = {
        "rf_default": "random_forest_model.py",
        "gb_default": "gradient_boosting_model.py",
        "lr_default": "logistic_regression_model.py",
        "lstm_default": "lstm_model.py"
    }
    
    if model_id not in model_file_map:
        raise HTTPException(status_code=404, detail="Model not found")
    
    filename = model_file_map[model_id]
    model_path = PREBUILT_MODELS_FOLDER / filename
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")
    
    return FileResponse(
        path=str(model_path),
        filename=filename,
        media_type='text/x-python'
    )

@api_router.get("/models/custom")
async def get_custom_models():
    """Get list of custom models"""
    try:
        models = await db.custom_models.find({}, {"_id": 0}).to_list(100)
        return {"models": models}
    except Exception as e:
        logger.error(f"MongoDB error in get_custom_models: {e}")
        return {"models": []}

@api_router.post("/models/custom")
async def create_custom_model(config: CustomModelConfig):
    """Create a custom model configuration"""
    model_data = {
        "id": str(uuid.uuid4()),
        "name": config.name,
        "type": "custom",
        "layers": config.layers,
        "optimizer": config.optimizer,
        "loss": config.loss,
        "learning_rate": config.learning_rate,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await db.custom_models.insert_one({**model_data})
    return model_data

@api_router.get("/models/saved")
async def get_saved_models():
    """Get list of saved trained models with metadata (including .py files)"""
    models = []
    
    try:
        # Get metadata from MongoDB (prioritize .py files)
        metadata_list = await db.saved_models.find({}, {"_id": 0}).to_list(1000)
        metadata_map = {m["id"]: m for m in metadata_list}
        
        # First, check for .py files in saved models folder
        search_paths = [SAVED_MODELS_FOLDER]
        if FALLBACK_SAVED_MODELS_FOLDER and FALLBACK_SAVED_MODELS_FOLDER.exists():
            search_paths.append(FALLBACK_SAVED_MODELS_FOLDER)
            
        for path in search_paths:
            for file in path.glob("*_model.py"):
                try:
                    stat_info = file.stat()
                    session_id = file.stem.replace("_model", "")
                    
                    if any(m["id"] == session_id for m in models):
                        continue
                        
                    meta = metadata_map.get(session_id, {})
                    session_info = training_sessions.get(session_id, {})
                    
                    name = meta.get("name") or session_info.get("model_name") or file.stem
                    model_type = meta.get("model_type") or "unknown"
                    
                    models.append({
                        "id": session_id,
                        "path": str(file),
                        "name": name,
                        "type": model_type,
                        "size": stat_info.st_size,
                        "file_type": "py",
                        "target_column": meta.get("target_column"),
                        "feature_columns": meta.get("feature_columns", []),
                        "created_at": meta.get("created_at") or session_info.get("started_at") or datetime.fromtimestamp(stat_info.st_mtime).isoformat()
                    })
                except Exception:
                    continue
        
        # Also include .joblib files for backward compatibility
        model_path = Path(MODEL_FOLDER)
        if model_path.exists():
            for file in model_path.glob("*_model.joblib"):
                try:
                    session_id = file.stem.replace("_model", "")
                    # Skip if we already have a .py version
                    if any(m["id"] == session_id for m in models):
                        continue
                    
                    stat_info = file.stat()
                    meta = metadata_map.get(session_id, {})
                    session_info = training_sessions.get(session_id, {})
                    
                    name = meta.get("name") or session_info.get("model_name") or file.name
                    model_type = meta.get("model_type") or "unknown"
                    
                    models.append({
                        "id": session_id,
                        "path": str(file),
                        "name": name,
                        "type": model_type,
                        "size": stat_info.st_size,
                        "file_type": "joblib",
                        "target_column": meta.get("target_column"),
                        "feature_columns": meta.get("feature_columns", []),
                        "created_at": meta.get("created_at") or session_info.get("started_at") or datetime.fromtimestamp(stat_info.st_mtime).isoformat()
                    })
                except Exception:
                    continue
    except Exception as e:
        logger.error(f"Error fetching saved models: {e}")
        # Fallback to just scanning the folder if MongoDB fails
        model_path = Path(MODEL_FOLDER)
        if model_path.exists():
            for file in model_path.glob("*_model.joblib"):
                try:
                    stat_info = file.stat()
                    session_id = file.stem.replace("_model", "")
                    models.append({
                        "id": session_id,
                        "path": str(file),
                        "name": file.name,
                        "size": stat_info.st_size,
                        "file_type": "joblib",
                        "created_at": datetime.fromtimestamp(stat_info.st_mtime).isoformat()
                    })
                except Exception:
                    continue
    
    # Sort by created_at descending
    models.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"models": models}

@api_router.delete("/models/saved/{model_id}")
async def delete_saved_model(model_id: str):
    """Delete a saved model and its metadata"""
    try:
        # Delete from filesystem
        model_path = os.path.join(MODEL_FOLDER, f"{model_id}_model.joblib")
        scaler_path = os.path.join(MODEL_FOLDER, f"{model_id}_scaler.joblib")
        
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(scaler_path):
            os.remove(scaler_path)
            
        # Delete from MongoDB
        await db.saved_models.delete_one({"id": model_id})
        await db.training_sessions.delete_one({"id": model_id})
        
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.delete("/models/custom/{model_id}")
async def delete_custom_model(model_id: str):
    """Delete a custom model configuration"""
    try:
        await db.custom_models.delete_one({"id": model_id})
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@api_router.post("/models/import")
async def import_model(folder: FolderPath):
    """Import model from folder"""
    path = Path(folder.path)
    imported = []
    
    for file in path.glob("*.joblib"):
        dest = os.path.join(MODEL_FOLDER, file.name)
        import shutil
        shutil.copy(file, dest)
        imported.append({"name": file.name, "path": dest})
    
    return {"imported": imported}

@api_router.post("/models/export")
async def export_model(model_id: str, destination: str):
    """Export model to destination folder"""
    source = os.path.join(MODEL_FOLDER, f"{model_id}_model.joblib")
    if not os.path.exists(source):
        raise HTTPException(status_code=404, detail="Model not found")
    
    import shutil
    dest_path = Path(destination)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    dest_file = dest_path / f"{model_id}_model.joblib"
    shutil.copy(source, dest_file)
    
    return {"success": True, "path": str(dest_file)}

@api_router.post("/models/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload model file (.joblib, .h5, .keras)"""
    try:
        # Validate file extension
        allowed_extensions = ['.joblib', '.h5', '.keras']
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Generate unique filename
        model_id = str(uuid.uuid4())
        filename = f"{model_id}_model{file_ext}"
        file_path = os.path.join(MODEL_FOLDER, filename)
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Store metadata in MongoDB
        model_metadata = {
            "id": model_id,
            "name": file.filename,
            "type": "uploaded",
            "file_path": file_path,
            "file_type": file_ext,
            "size": len(content),
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        await db.uploaded_models.insert_one({**model_metadata})
        
        return {
            "success": True, 
            "model_id": model_id,
            "filename": filename,
            "path": file_path,
            "type": file_ext
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model upload error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Training Routes
@api_router.post("/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a training session"""
    session_id = str(uuid.uuid4())
    
    # Load model class from .py file
    model_instance = None
    try:
        model_class = load_model_class(request.model_id)
        # Get parameters from pre-built models list
        prebuilt_models = get_prebuilt_models()
        model_config = next((m for m in prebuilt_models if m["id"] == request.model_id), None)
        if model_config:
            model_instance = model_class(parameters=model_config.get("parameters", {}))
        else:
            # Fallback for custom models
            custom_model = await db.custom_models.find_one({"id": request.model_id}, {"_id": 0})
            if custom_model:
                # For custom models, use default parameters
                model_instance = model_class()
            else:
                raise HTTPException(status_code=404, detail="Model not found")
    except (ValueError, FileNotFoundError) as e:
        # Try custom model fallback
        custom_model = await db.custom_models.find_one({"id": request.model_id}, {"_id": 0})
        if custom_model:
            # For custom models, we'll use the old method for now
            model_config = custom_model
            model_instance = None
        else:
            raise HTTPException(status_code=404, detail=f"Model not found: {str(e)}")
    
    # Set default model name if not provided
    model_name = request.model_name
    if not model_name and model_config:
        model_name = model_config.get("name", "Unnamed Model")
    
    # Initialize session
    training_sessions[session_id] = {
        "id": session_id,
        "model_id": request.model_id,
        "model_name": model_name,
        "status": "pending",
        "current_epoch": 0,
        "total_epochs": request.epochs,
        "metrics": {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []},
        "started_at": None,
        "completed_at": None
    }
    
    log_queues[session_id] = queue.Queue()
    output_queues[session_id] = queue.Queue()
    
    # Start training in background
    if model_instance:
        asyncio.create_task(run_training(session_id, request, model_instance))
    else:
        # Fallback for custom models (old method)
        raise HTTPException(status_code=400, detail="Custom models not yet supported with new model class system")
    
    return {"session_id": session_id, "status": "started"}

@api_router.get("/training/status/{session_id}")
async def get_training_status(session_id: str):
    """Get training session status"""
    if session_id not in training_sessions:
        # Check MongoDB
        try:
            session = await db.training_sessions.find_one({"id": session_id}, {"_id": 0})
            if session:
                return session
        except Exception as e:
            logger.error(f"MongoDB status check failed: {e}")
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Return session without non-serializable fields
    session_data = training_sessions[session_id].copy()
    if "model_instance" in session_data:
        del session_data["model_instance"]
    return session_data

@api_router.post("/training/stop/{session_id}")
async def stop_training(session_id: str):
    """Stop a training session"""
    if session_id not in training_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    training_sessions[session_id]["status"] = "stopped"
    return {"success": True, "status": "stopped"}

@api_router.get("/training/logs/{session_id}")
async def get_training_logs(session_id: str):
    """Get training logs"""
    logs = []
    if session_id in log_queues:
        while not log_queues[session_id].empty():
            try:
                logs.append(log_queues[session_id].get_nowait())
            except:
                break
    return {"logs": logs}

@api_router.get("/training/output/{session_id}")
async def get_model_output(session_id: str):
    """Get model output (predictions)"""
    outputs = []
    if session_id in output_queues:
        while not output_queues[session_id].empty():
            try:
                outputs.append(output_queues[session_id].get_nowait())
            except:
                break
    return {"outputs": outputs}

@api_router.post("/training/save/{session_id}")
async def save_trained_model(session_id: str, save_request: Optional[SaveModelRequest] = None):
    """Save a trained model after training completes"""
    if session_id not in training_sessions:
        # Check MongoDB
        session = await db.training_sessions.find_one({"id": session_id}, {"_id": 0})
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        training_sessions[session_id] = session
    
    session = training_sessions[session_id]
    
    if session.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Model training must be completed before saving")
    
    # Check if we have the model instance in memory
    model_instance = model_instances.get(session_id)
    if not model_instance:
        raise HTTPException(status_code=400, detail="Model instance not found in memory. It may have been lost if the server restarted.")
    
    try:
        # Prioritize name from save request, then session name, then default
        model_name = (save_request.name if save_request and save_request.name else None) or session.get("model_name") or f"Model_{session_id[:8]}"
        target_column = session.get("target_column")
        feature_columns = session.get("feature_columns", [])
        
        # Save as .joblib files
        model_path = os.path.join(MODEL_FOLDER, f"{session_id}_model.joblib")
        scaler_path = os.path.join(MODEL_FOLDER, f"{session_id}_scaler.joblib")
        joblib.dump(model_instance.model, model_path)
        if model_instance.scaler:
            joblib.dump(model_instance.scaler, scaler_path)
        
        # Save as .py file
        py_model_path = SAVED_MODELS_FOLDER / f"{session_id}_model.py"
        
        # Generate Python file content
        model_type_map = {
            "random_forest": ("RandomForestModel", "random_forest_model"),
            "gradient_boosting": ("GradientBoostingModel", "gradient_boosting_model"),
            "logistic_regression": ("LogisticRegressionModel", "logistic_regression_model"),
            "lstm": ("LSTMModel", "lstm_model")
        }
        
        model_type = model_instance.MODEL_TYPE
        class_name, module_name = model_type_map.get(model_type, ("RandomForestModel", "random_forest_model"))
        
        py_content = f'''"""
Trained {model_instance.MODEL_NAME}
Saved on {datetime.now(timezone.utc).isoformat()}
"""
import joblib
from pathlib import Path
from models.prebuilt.{module_name} import {class_name}

# Initialize model instance
model = {class_name}(parameters={model_instance.parameters})

# Load trained model weights
model_path = Path(__file__).parent / "{session_id}_model.joblib"
scaler_path = Path(__file__).parent / "{session_id}_scaler.joblib"

model.model = joblib.load(model_path)
if scaler_path.exists():
    model.scaler = joblib.load(scaler_path)

# Set column configuration
model.TARGET_COLUMN = "{target_column}"
model.FEATURE_COLUMNS = {feature_columns}
'''
        
        # Write .py file
        with open(py_model_path, 'w') as f:
            f.write(py_content)
        
        # Also copy .joblib files to saved folder for reference
        import shutil
        saved_model_path = SAVED_MODELS_FOLDER / f"{session_id}_model.joblib"
        saved_scaler_path = SAVED_MODELS_FOLDER / f"{session_id}_scaler.joblib"
        shutil.copy(model_path, saved_model_path)
        if os.path.exists(scaler_path):
            shutil.copy(scaler_path, saved_scaler_path)
        
        # Store metadata in MongoDB
        await db.saved_models.insert_one({
            "id": session_id,
            "name": model_name,
            "model_type": model_type,
            "file_path": str(py_model_path),
            "joblib_path": str(saved_model_path),
            "target_column": target_column,
            "feature_columns": feature_columns,
            "created_at": datetime.now(timezone.utc).isoformat()
        })
        
        return {
            "success": True,
            "session_id": session_id,
            "model_name": model_name,
            "py_file": str(py_model_path),
            "joblib_file": str(saved_model_path)
        }
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save model: {str(e)}")

@api_router.get("/training/sessions")
async def get_all_sessions():
    """Get all training sessions with full metrics"""
    sessions = list(training_sessions.values())
    try:
        # Get all sessions from DB with full metrics
        db_sessions = await db.training_sessions.find(
            {}, 
            {"_id": 0}
        ).sort("started_at", -1).limit(100).to_list(100)
        
        # Merge, avoiding duplicates - prioritize in-memory sessions (more current)
        session_ids = {s["id"] for s in sessions}
        for db_session in db_sessions:
            if db_session["id"] not in session_ids:
                sessions.append(db_session)
        
        # Sort: running first, then by started_at descending
        def sort_key(s):
            status_priority = {"running": 0, "pending": 1, "completed": 2, "failed": 3, "stopped": 4}
            return (
                status_priority.get(s.get("status", "completed"), 5),
                -(datetime.fromisoformat(s.get("started_at", "1970-01-01T00:00:00+00:00").replace("Z", "+00:00")).timestamp() if s.get("started_at") else 0)
            )
        
        sessions.sort(key=sort_key)
        
    except Exception as e:
        logger.error(f"MongoDB error in get_all_sessions: {e}")
        # Continue with in-memory sessions only
    
    return {"sessions": sessions}

# Backtesting Routes
@api_router.post("/backtest/run")
async def run_backtest(request: BacktestRequest):
    """Run backtesting on a trained model using .py model file"""
    # Try to load from saved models metadata first
    saved_model = None
    try:
        saved_model = await db.saved_models.find_one({"id": request.model_id}, {"_id": 0})
    except Exception as db_error:
        logger.warning(f"Failed to fetch model metadata from MongoDB: {db_error}. Falling back to file-based lookup.")
    
    model_instance = None
    target_column = None
    feature_columns = None
    
    # Try to find and load .py file (either from DB metadata or direct check)
    py_file_path = None
    if saved_model and saved_model.get("file_path"):
        py_file_path = Path(saved_model["file_path"])
    
    # If not found or doesn't exist, check local current folder
    if not py_file_path or not py_file_path.exists():
        py_file_path = SAVED_MODELS_FOLDER / f"{request.model_id}_model.py"
    
    # If still not found, check fallback folder
    if not py_file_path.exists() and FALLBACK_SAVED_MODELS_FOLDER:
        py_file_path = FALLBACK_SAVED_MODELS_FOLDER / f"{request.model_id}_model.py"
    
    if py_file_path and py_file_path.exists():
        # Load from .py file
        try:
            # Use absolute path for importlib
            py_file_path = py_file_path.absolute()
            spec = importlib.util.spec_from_file_location(f"backtest_model_{request.model_id}", py_file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"backtest_model_{request.model_id}"] = module
            spec.loader.exec_module(module)
            
            # Get model instance from module
            model_instance = module.model
            target_column = getattr(model_instance, 'TARGET_COLUMN', None)
            feature_columns = getattr(model_instance, 'FEATURE_COLUMNS', [])
        except Exception as e:
            logger.error(f"Failed to load model from .py file: {e}")
            # Don't raise yet, try .joblib fallback
    
    # Fallback to .joblib if .py loading failed
    if model_instance is None:
        model_path = os.path.join(MODEL_FOLDER, f"{request.model_id}_model.joblib")
        scaler_path = os.path.join(MODEL_FOLDER, f"{request.model_id}_scaler.joblib")
        
        if not os.path.exists(model_path):
            # Try saved models folder
            saved_model_path = SAVED_MODELS_FOLDER / f"{request.model_id}_model.joblib"
            saved_scaler_path = SAVED_MODELS_FOLDER / f"{request.model_id}_scaler.joblib"
            if saved_model_path.exists():
                model_path = str(saved_model_path)
                scaler_path = str(saved_scaler_path) if saved_scaler_path.exists() else None
        
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")
        
        # For backward compatibility, use old method
        # But we still need target_column and feature_columns
        if saved_model:
            target_column = saved_model.get("target_column")
            feature_columns = saved_model.get("feature_columns", [])
        else:
            raise HTTPException(status_code=400, detail="Model metadata not found. Please use a saved model with .py file.")
    
    try:
        # Load test data
        df = pd.read_csv(request.test_data_path)
        
        # Use model's column configuration
        if model_instance:
            # Use model's test method
            if target_column and feature_columns:
                # Prepare features using model's columns
                missing_features = [col for col in feature_columns if col not in df.columns]
                if missing_features:
                    raise ValueError(f"Feature columns not found in data: {missing_features}")
                
                X_df = df[feature_columns].copy()
                # Drop non-numeric columns
                non_numeric_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
                if non_numeric_cols:
                    X_df = X_df.drop(columns=non_numeric_cols)
                
                X = X_df.values
                
                # Scale if scaler exists
                if model_instance.scaler:
                    X = model_instance.scaler.transform(X)
                
                # Use model's test method
                predictions, probabilities = model_instance.test(X)
            else:
                raise ValueError("Model missing target_column or feature_columns configuration")
        else:
            # Fallback: load .joblib and use old method
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None
            
            if not target_column or not feature_columns:
                raise ValueError("Model metadata missing. Cannot determine target and feature columns.")
            
            X = df[feature_columns].values
            if scaler:
                X = scaler.transform(X)
            
            predictions = model.predict(X)
            
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(X)
                    if proba.shape[1] == 2:
                        probabilities = proba[:, 1]
                    else:
                        probabilities = proba[:, 0]
                except Exception:
                    probabilities = predictions.astype(float)
            else:
                probabilities = predictions.astype(float)
        
        # Prepare case-insensitive column map
        lowered_cols = {col.lower(): col for col in df.columns}
        
        # Helper to get column name regardless of case
        def get_col(candidates):
            for c in candidates:
                if c.lower() in lowered_cols:
                    return lowered_cols[c.lower()]
            return None

        # Detect columns
        col_open = get_col(['open', 'Open'])
        col_high = get_col(['high', 'High'])
        col_low = get_col(['low', 'Low'])
        col_close = get_col(['close', 'Close'])
        col_date = get_col(['date', 'Date', 'DateTime', 'Timestamp'])
        
        # Check for OHLC columns
        has_ohlc = all([col_open, col_high, col_low, col_close])
        
        # Get price data
        if has_ohlc:
            opens = df[col_open].values
            highs = df[col_high].values
            lows = df[col_low].values
            closes = df[col_close].values
        elif col_close:
            closes = df[col_close].values
            # Generate synthetic OHLC relative to actual close
            opens = closes * (1 - np.random.random(len(closes)) * 0.002)
            highs = closes * (1 + np.random.random(len(closes)) * 0.005)
            lows = closes * (1 - np.random.random(len(closes)) * 0.005)
        elif 'price' in lowered_cols:
            col_price = lowered_cols['price']
            closes = df[col_price].values
            opens = closes * 0.998
            highs = closes * 1.005
            lows = closes * 0.995
        else:
            # Generate synthetic prices only if no price columns found
            logger.warning("No price columns found. Generating synthetic data.")
            closes = np.cumsum(np.random.randn(len(df)) * 0.02 + 0.001) + 100
            opens = closes * 0.998
            highs = closes * 1.005
            lows = closes * 0.995
        
        # Prepare timestamps for visualization and market hour checking
        timestamps = []
        dt_timestamps = None
        
        if col_date:
            try:
                # Use mixed format for flexibility
                dt_series = pd.to_datetime(df[col_date], format='mixed', errors='coerce')
                dt_timestamps = dt_series
                
                # Create string versions, falling back to original string for unparseable dates
                formatted_dates = dt_series.dt.strftime('%Y-%m-%d %H:%M:%S')
                orig_strings = df[col_date].astype(str).values
                
                timestamps = [
                    formatted_dates.iloc[i] if pd.notnull(formatted_dates.iloc[i]) else orig_strings[i]
                    for i in range(len(df))
                ]
            except Exception as e:
                logger.warning(f"Failed to parse date column: {e}. Falling back to strings.")
                timestamps = df[col_date].astype(str).values
        else:
            timestamps = [f"Point_{i}" for i in range(len(df))]

        # Simulate trading with candle targeting or RR ratio
        capital = request.initial_capital
        position = 0
        trades = []
        equity_curve = [capital]
        target_candle = max(1, min(request.target_candle, 10))  # Limit to 1-10 candles
        
        # Risk management parameters
        rr_ratio = request.rr_ratio
        current_tp = None
        current_sl = None
        
        price_data = []
        for i in range(len(df)):
            timestamp = timestamps[i]
            pred = predictions[i]
            prob = probabilities[i]
            
            # Check for market hours (09:30 - 16:00)
            is_market_hours = True
            if dt_timestamps is not None:
                try:
                    current_dt = dt_timestamps.iloc[i]
                    # Ensure current_dt is a timestamp object with time
                    if hasattr(current_dt, 'hour'):
                        market_start = current_dt.replace(hour=9, minute=30, second=0, microsecond=0)
                        market_end = current_dt.replace(hour=16, minute=0, second=0, microsecond=0)
                        is_market_hours = (current_dt >= market_start) and (current_dt <= market_end)
                    else:
                        is_market_hours = True
                except Exception as e:
                    is_market_hours = True

            # Trading logic with candle targeting or RR
            # Only enter if within market hours
            if pred == 1 and position == 0 and i < len(df) - 1 and is_market_hours:  # Buy signal
                entry_price = closes[i]
                target_position_size = capital * request.position_size
                
                # Round shares to whole number
                shares = round(target_position_size / entry_price)
                
                if shares > 0:
                    actual_cost = shares * entry_price
                    # Add commission on buy
                    total_entry_cost = actual_cost + request.commission_fee
                    
                    if total_entry_cost <= capital:
                        position = shares
                        capital -= total_entry_cost
                        
                        trade_info = {
                            "type": "BUY",
                            "time": timestamp,
                            "price": float(entry_price),
                            "shares": int(shares),
                            "value": float(actual_cost),
                            "fee": float(request.commission_fee),
                            "probability": float(prob),
                        }

                        if rr_ratio:
                            # Use a default risk of 1% for SL
                            risk_pct = 0.01 
                            current_sl = entry_price * (1 - risk_pct)
                            current_tp = entry_price + (entry_price - current_sl) * rr_ratio
                            trade_info["tp"] = float(current_tp)
                            trade_info["sl"] = float(current_sl)
                        else:
                            # Original logic: Exit after N candles
                            target_idx = min(i + target_candle, len(df) - 1)
                            trade_info["target_candle"] = target_candle
                            trade_info["target_idx"] = target_idx
                            current_tp = None
                            current_sl = None
                        
                        trades.append(trade_info)
                
            elif position > 0:
                last_buy = [t for t in trades if t["type"] == "BUY"][-1]
                exit_signal = False
                exit_price = closes[i]
                is_hit = False

                if rr_ratio:
                    # RR Logic: Close on TP or SL hit
                    if highs[i] >= current_tp:
                        exit_signal = True
                        exit_price = current_tp
                        is_hit = True
                    elif lows[i] <= current_sl:
                        exit_signal = True
                        exit_price = current_sl
                        is_hit = False
                    # Force close on last candle if still open
                    elif i == len(df) - 1:
                        exit_signal = True
                        exit_price = closes[i]
                        is_hit = False
                else:
                    # Original logic: Exit at target candle
                    if i >= last_buy.get("target_idx", i + 1):
                        exit_signal = True
                        exit_price = closes[i]
                        is_hit = True
                    # Force close on last candle if still open
                    elif i == len(df) - 1:
                        exit_signal = True
                        exit_price = closes[i]
                        is_hit = False
                
                if exit_signal:
                    gross_value = position * exit_price
                    # Subtract commission on sell
                    net_value = gross_value - request.commission_fee
                    capital += net_value
                    
                    # P&L is net proceeds - (original cost + buy fee)
                    pnl = net_value - (last_buy["value"] + last_buy["fee"])
                    
                    trades.append({
                        "type": "SELL",
                        "time": timestamp,
                        "price": float(exit_price),
                        "shares": int(position),
                        "value": float(gross_value),
                        "fee": float(request.commission_fee),
                        "pnl": float(pnl),
                        "probability": float(prob),
                        "is_hit": is_hit
                    })
                    position = 0
                    current_tp = None
                    current_sl = None

            # Track equity
            # Account for unrealized exit fee when position is open for a smoother curve
            exit_fee_allowance = request.commission_fee if position > 0 else 0
            current_equity = capital + (position * closes[i] if position > 0 else 0) - exit_fee_allowance
            equity_curve.append(current_equity)
            
            price_data.append({
                "time": timestamp,
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "prediction": int(pred),
                "probability": float(prob),
                "equity": float(current_equity),
                "tp": float(current_tp) if current_tp else None,
                "sl": float(current_sl) if current_sl else None,
                "is_market_hours": is_market_hours
            })
        
        # Calculate metrics
        total_return = (equity_curve[-1] - request.initial_capital) / request.initial_capital * 100
        winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
        losing_trades = sum(1 for t in trades if t.get("pnl", 0) < 0)
        
        # Calculate max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            max_dd = max(max_dd, dd)
        
        # Calculate Sharpe ratio (simplified)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)
        
        result = BacktestResult(
            model_id=request.model_id,
            total_trades=len([t for t in trades if t["type"] == "SELL"]),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_return=float(total_return),
            sharpe_ratio=float(sharpe),
            max_drawdown=float(max_dd),
            trades=trades,
            price_data=price_data
        )
        
        # Store in MongoDB (handle connection errors)
        try:
            await db.backtest_results.insert_one({**result.model_dump()})
        except Exception as db_error:
            logger.error(f"Failed to save backtest result to MongoDB: {db_error}")
        
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/backtest/results")
async def get_backtest_results():
    """Get all backtest results"""
    try:
        # Only return essential fields for list view - exclude large fields like trades and price_data
        projection = {
            "_id": 0,
            "id": 1,
            "model_id": 1,
            "total_trades": 1,
            "winning_trades": 1,
            "losing_trades": 1,
            "total_return": 1,
            "sharpe_ratio": 1,
            "max_drawdown": 1,
            "created_at": 1
        }
        results = await db.backtest_results.find({}, projection).sort("created_at", -1).limit(100).to_list(100)
        return {"results": results}
    except Exception as e:
        logger.error(f"MongoDB error in get_backtest_results: {e}")
        return {"results": []}

@api_router.get("/backtest/result/{result_id}")
async def get_backtest_result(result_id: str):
    """Get specific backtest result"""
    try:
        result = await db.backtest_results.find_one({"id": result_id}, {"_id": 0})
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MongoDB error in get_backtest_result: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")

# Dashboard Stats
@api_router.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    total_models = len(PREBUILT_MODELS)
    
    # Default values in case MongoDB is unavailable
    custom_models = 0
    training_sessions_db = 0
    backtest_count = 0
    latest_backtest = None
    
    try:
        # Run MongoDB queries in parallel for better performance
        custom_models_task = db.custom_models.count_documents({})
        training_sessions_db_task = db.training_sessions.count_documents({})
        backtest_count_task = db.backtest_results.count_documents({})
        
        # Get latest backtest with limited fields (exclude large data)
        latest_backtest_task = db.backtest_results.find_one(
            {}, 
            {
                "_id": 0,
                "id": 1,
                "model_id": 1,
                "total_trades": 1,
                "winning_trades": 1,
                "losing_trades": 1,
                "total_return": 1,
                "sharpe_ratio": 1,
                "max_drawdown": 1,
                "created_at": 1
            },
            sort=[("created_at", -1)]
        )
        
        # Wait for all queries to complete
        custom_models, training_sessions_db, backtest_count, latest_backtest = await asyncio.gather(
            custom_models_task,
            training_sessions_db_task,
            backtest_count_task,
            latest_backtest_task,
            return_exceptions=True
        )
        
        # Handle exceptions from gather
        if isinstance(custom_models, Exception):
            logger.warning(f"MongoDB error getting custom_models: {custom_models}")
            custom_models = 0
        if isinstance(training_sessions_db, Exception):
            logger.warning(f"MongoDB error getting training_sessions: {training_sessions_db}")
            training_sessions_db = 0
        if isinstance(backtest_count, Exception):
            logger.warning(f"MongoDB error getting backtest_count: {backtest_count}")
            backtest_count = 0
        if isinstance(latest_backtest, Exception):
            logger.warning(f"MongoDB error getting latest_backtest: {latest_backtest}")
            latest_backtest = None
            
    except Exception as e:
        logger.error(f"MongoDB connection error in get_dashboard_stats: {e}")
        # Continue with default values
    
    training_sessions_count = len(training_sessions) + training_sessions_db
    
    # Get latest training metrics
    latest_session = None
    if training_sessions:
        latest_session = list(training_sessions.values())[-1]
    
    return {
        "total_models": total_models + custom_models,
        "custom_models": custom_models,
        "training_sessions": training_sessions_count,
        "backtest_count": backtest_count,
        "latest_session": latest_session,
        "latest_backtest": latest_backtest,
        "active_trainings": sum(1 for s in training_sessions.values() if s["status"] == "running")
    }

@api_router.get("/dashboard/model-performance")
async def get_model_performance():
    """Get performance metrics for all models"""
    try:
        # Aggregate backtest results by model_id
        pipeline = [
            {
                "$group": {
                    "_id": "$model_id",
                    "total_backtests": {"$sum": 1},
                    "avg_return": {"$avg": "$total_return"},
                    "max_return": {"$max": "$total_return"},
                    "min_return": {"$min": "$total_return"},
                    "avg_sharpe": {"$avg": "$sharpe_ratio"},
                    "total_trades": {"$sum": "$total_trades"},
                    "winning_trades": {"$sum": "$winning_trades"},
                    "losing_trades": {"$sum": "$losing_trades"},
                    "last_backtest": {"$max": "$created_at"}
                }
            },
            {"$sort": {"avg_return": -1}}
        ]
        
        results = await db.backtest_results.aggregate(pipeline).to_list(100)
        
        # Format results
        performance_data = []
        for result in results:
            performance_data.append({
                "model_id": result["_id"],
                "total_backtests": result["total_backtests"],
                "avg_return": round(result["avg_return"], 2),
                "max_return": round(result["max_return"], 2),
                "min_return": round(result["min_return"], 2),
                "avg_sharpe": round(result["avg_sharpe"], 2),
                "total_trades": result["total_trades"],
                "winning_trades": result["winning_trades"],
                "losing_trades": result["losing_trades"],
                "win_rate": round((result["winning_trades"] / max(1, result["total_trades"])) * 100, 1),
                "last_backtest": result["last_backtest"]
            })
        
        return {"performance": performance_data}
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        return {"performance": []}

# WebSocket for real-time updates
@api_router.websocket("/ws/training/{session_id}")
async def training_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in websocket_connections:
        websocket_connections[session_id] = []
    websocket_connections[session_id].append(websocket)
    
    try:
        while True:
            # Send updates
            if session_id in training_sessions:
                await websocket.send_json({
                    "type": "status",
                    "data": training_sessions[session_id]
                })
            
            # Send logs
            if session_id in log_queues:
                logs = []
                while not log_queues[session_id].empty():
                    logs.append(log_queues[session_id].get_nowait())
                if logs:
                    await websocket.send_json({"type": "logs", "data": logs})
            
            # Send outputs
            if session_id in output_queues:
                outputs = []
                while not output_queues[session_id].empty():
                    outputs.append(output_queues[session_id].get_nowait())
                if outputs:
                    await websocket.send_json({"type": "output", "data": outputs})
            
            await asyncio.sleep(0.5)
            
    except WebSocketDisconnect:
        websocket_connections[session_id].remove(websocket)

# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
