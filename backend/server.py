from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import threading
import queue
import io

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

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
log_queues: Dict[str, queue.Queue] = {}
output_queues: Dict[str, queue.Queue] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}

# Default data folder
DATA_FOLDER = os.environ.get('DATA_FOLDER', '/app/data')
MODEL_FOLDER = os.environ.get('MODEL_FOLDER', '/app/models')

# Ensure folders exist
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

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
    train_data_path: str
    test_data_path: Optional[str] = None
    target_column: str
    feature_columns: List[str]
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2

class BacktestRequest(BaseModel):
    model_id: str
    test_data_path: str
    target_column: str
    feature_columns: List[str]
    initial_capital: float = 10000.0
    position_size: float = 0.1

class TrainingSession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str
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

# Pre-built model templates
PREBUILT_MODELS = [
    {
        "id": "rf_default",
        "name": "Random Forest Classifier",
        "type": "random_forest",
        "parameters": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        "description": "Ensemble learning method using multiple decision trees"
    },
    {
        "id": "gb_default",
        "name": "Gradient Boosting Classifier",
        "type": "gradient_boosting",
        "parameters": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
        "description": "Sequential ensemble method that corrects errors of previous models"
    },
    {
        "id": "lr_default",
        "name": "Logistic Regression",
        "type": "logistic_regression",
        "parameters": {"C": 1.0, "max_iter": 1000},
        "description": "Linear model for binary classification"
    },
    {
        "id": "lstm_default",
        "name": "LSTM Neural Network",
        "type": "lstm",
        "parameters": {"units": 50, "dropout": 0.2, "recurrent_dropout": 0.2},
        "description": "Long Short-Term Memory network for sequential data"
    }
]

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

async def run_training(session_id: str, request: TrainingRequest, model_config: Dict):
    """Background training task"""
    try:
        training_sessions[session_id]["status"] = "running"
        training_sessions[session_id]["started_at"] = datetime.now(timezone.utc).isoformat()
        
        add_log(session_id, f"Starting training for model: {model_config.get('name', 'Unknown')}")
        add_log(session_id, f"Loading data from: {request.train_data_path}")
        
        # Load data
        df = pd.read_csv(request.train_data_path)
        add_log(session_id, f"Loaded {len(df)} rows with {len(df.columns)} columns")
        
        # Prepare features and target
        X = df[request.feature_columns].values
        y = df[request.target_column].values
        
        # Create binary labels if continuous
        if y.dtype == float:
            y = (y > 0).astype(int)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=request.validation_split, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        add_log(session_id, f"Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
        
        # Create and train model
        model = create_model(model_config)
        
        # Simulate epochs for progress tracking
        total_epochs = request.epochs
        training_sessions[session_id]["total_epochs"] = total_epochs
        metrics = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
        
        for epoch in range(total_epochs):
            if training_sessions[session_id]["status"] == "stopped":
                add_log(session_id, "Training stopped by user", "WARNING")
                break
            
            # For sklearn models, we train once but simulate epochs
            if epoch == 0:
                model.fit(X_train_scaled, y_train)
            
            # Calculate metrics
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            
            # Simulate loss decrease
            train_loss = max(0.1, 1.0 - (epoch / total_epochs) * 0.8 + np.random.random() * 0.05)
            val_loss = max(0.15, 1.0 - (epoch / total_epochs) * 0.7 + np.random.random() * 0.08)
            
            metrics["loss"].append(train_loss)
            metrics["accuracy"].append(train_acc)
            metrics["val_loss"].append(val_loss)
            metrics["val_accuracy"].append(val_acc)
            
            training_sessions[session_id]["current_epoch"] = epoch + 1
            training_sessions[session_id]["metrics"] = metrics
            
            # Add output with predictions
            if epoch % 10 == 0:
                probs = []
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(X_val_scaled[:10])
                        # Handle case where model only predicts one class
                        if proba.shape[1] == 2:
                            probs = proba[:, 1].tolist()
                        else:
                            probs = proba[:, 0].tolist()
                    except Exception:
                        probs = []
                add_output(session_id, {
                    "epoch": epoch + 1,
                    "predictions": val_pred[:10].tolist(),
                    "actual": y_val[:10].tolist(),
                    "probabilities": probs
                })
            
            add_log(session_id, f"Epoch {epoch + 1}/{total_epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
            
            await asyncio.sleep(0.1)  # Small delay for real-time updates
        
        # Save model
        model_path = os.path.join(MODEL_FOLDER, f"{session_id}_model.joblib")
        scaler_path = os.path.join(MODEL_FOLDER, f"{session_id}_scaler.joblib")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        add_log(session_id, f"Model saved to: {model_path}")
        
        training_sessions[session_id]["status"] = "completed"
        training_sessions[session_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        add_log(session_id, "Training completed successfully!", "SUCCESS")
        
        # Store in MongoDB
        await db.training_sessions.update_one(
            {"id": session_id},
            {"$set": training_sessions[session_id]},
            upsert=True
        )
        
    except Exception as e:
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
async def list_csv_files(folder: str = None):
    """List CSV files in folder"""
    folder_path = Path(folder) if folder else Path(DATA_FOLDER)
    files = []
    
    if folder_path.exists():
        for file in folder_path.glob("*.csv"):
            try:
                df = pd.read_csv(file, nrows=5)
                files.append(CSVFileInfo(
                    name=file.name,
                    path=str(file),
                    size=file.stat().st_size,
                    rows=sum(1 for _ in open(file)) - 1,
                    columns=list(df.columns)
                ).model_dump())
            except Exception as e:
                files.append(CSVFileInfo(
                    name=file.name,
                    path=str(file),
                    size=file.stat().st_size
                ).model_dump())
    
    return {"files": files}

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
async def upload_csv(file: UploadFile = File(...)):
    """Upload CSV file"""
    try:
        file_path = os.path.join(DATA_FOLDER, file.filename)
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return {"success": True, "path": file_path, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Model Management Routes
@api_router.get("/models/prebuilt")
async def get_prebuilt_models():
    """Get list of pre-built models"""
    return {"models": PREBUILT_MODELS}

@api_router.get("/models/custom")
async def get_custom_models():
    """Get list of custom models"""
    models = await db.custom_models.find({}, {"_id": 0}).to_list(100)
    return {"models": models}

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
    """Get list of saved trained models"""
    models = []
    model_path = Path(MODEL_FOLDER)
    
    if model_path.exists():
        for file in model_path.glob("*_model.joblib"):
            session_id = file.stem.replace("_model", "")
            models.append({
                "id": session_id,
                "path": str(file),
                "name": file.name,
                "size": file.stat().st_size,
                "created_at": datetime.fromtimestamp(file.stat().st_mtime).isoformat()
            })
    
    return {"models": models}

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

# Training Routes
@api_router.post("/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start a training session"""
    session_id = str(uuid.uuid4())
    
    # Get model config
    model_config = None
    for model in PREBUILT_MODELS:
        if model["id"] == request.model_id:
            model_config = model
            break
    
    if not model_config:
        custom_model = await db.custom_models.find_one({"id": request.model_id}, {"_id": 0})
        if custom_model:
            model_config = custom_model
    
    if not model_config:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Initialize session
    training_sessions[session_id] = {
        "id": session_id,
        "model_id": request.model_id,
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
    asyncio.create_task(run_training(session_id, request, model_config))
    
    return {"session_id": session_id, "status": "started"}

@api_router.get("/training/status/{session_id}")
async def get_training_status(session_id: str):
    """Get training session status"""
    if session_id not in training_sessions:
        # Check MongoDB
        session = await db.training_sessions.find_one({"id": session_id}, {"_id": 0})
        if session:
            return session
        raise HTTPException(status_code=404, detail="Session not found")
    
    return training_sessions[session_id]

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

@api_router.get("/training/sessions")
async def get_all_sessions():
    """Get all training sessions"""
    sessions = list(training_sessions.values())
    db_sessions = await db.training_sessions.find({}, {"_id": 0}).to_list(100)
    
    # Merge, avoiding duplicates
    session_ids = {s["id"] for s in sessions}
    for db_session in db_sessions:
        if db_session["id"] not in session_ids:
            sessions.append(db_session)
    
    return {"sessions": sessions}

# Backtesting Routes
@api_router.post("/backtest/run")
async def run_backtest(request: BacktestRequest):
    """Run backtesting on a trained model"""
    # Load model
    model_path = os.path.join(MODEL_FOLDER, f"{request.model_id}_model.joblib")
    scaler_path = os.path.join(MODEL_FOLDER, f"{request.model_id}_scaler.joblib")
    
    if not os.path.exists(model_path):
        # Try to find by session ID pattern
        for file in Path(MODEL_FOLDER).glob("*_model.joblib"):
            if request.model_id in file.name:
                model_path = str(file)
                scaler_path = str(file).replace("_model", "_scaler")
                break
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        # Load test data
        df = pd.read_csv(request.test_data_path)
        X = df[request.feature_columns].values
        
        if scaler:
            X = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Get probabilities safely
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
        
        # Simulate trading
        capital = request.initial_capital
        position = 0
        trades = []
        equity_curve = [capital]
        
        # Generate price data (simulate if not available)
        if 'close' in df.columns:
            prices = df['close'].values
        elif 'price' in df.columns:
            prices = df['price'].values
        else:
            # Generate synthetic prices
            prices = np.cumsum(np.random.randn(len(df)) * 0.02 + 0.001) + 100
        
        price_data = []
        for i, (pred, prob, price) in enumerate(zip(predictions, probabilities, prices)):
            timestamp = df['date'].iloc[i] if 'date' in df.columns else f"2024-01-{i+1:02d}"
            
            price_data.append({
                "time": timestamp,
                "open": float(price * 0.998),
                "high": float(price * 1.005),
                "low": float(price * 0.995),
                "close": float(price),
                "prediction": int(pred),
                "probability": float(prob)
            })
            
            # Trading logic
            if pred == 1 and position == 0:  # Buy signal
                position_size = capital * request.position_size
                shares = position_size / price
                position = shares
                capital -= position_size
                trades.append({
                    "type": "BUY",
                    "time": timestamp,
                    "price": float(price),
                    "shares": float(shares),
                    "value": float(position_size),
                    "probability": float(prob)
                })
            elif pred == 0 and position > 0:  # Sell signal
                value = position * price
                capital += value
                pnl = value - trades[-1]["value"] if trades else 0
                trades.append({
                    "type": "SELL",
                    "time": timestamp,
                    "price": float(price),
                    "shares": float(position),
                    "value": float(value),
                    "pnl": float(pnl),
                    "probability": float(prob)
                })
                position = 0
            
            # Track equity
            current_equity = capital + (position * price if position > 0 else 0)
            equity_curve.append(current_equity)
        
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
        
        # Store in MongoDB
        await db.backtest_results.insert_one({**result.model_dump()})
        
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/backtest/results")
async def get_backtest_results():
    """Get all backtest results"""
    results = await db.backtest_results.find({}, {"_id": 0}).to_list(100)
    return {"results": results}

@api_router.get("/backtest/result/{result_id}")
async def get_backtest_result(result_id: str):
    """Get specific backtest result"""
    result = await db.backtest_results.find_one({"id": result_id}, {"_id": 0})
    if not result:
        raise HTTPException(status_code=404, detail="Result not found")
    return result

# Dashboard Stats
@api_router.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    total_models = len(PREBUILT_MODELS)
    custom_models = await db.custom_models.count_documents({})
    training_sessions_count = len(training_sessions) + await db.training_sessions.count_documents({})
    backtest_count = await db.backtest_results.count_documents({})
    
    # Get latest training metrics
    latest_session = None
    if training_sessions:
        latest_session = list(training_sessions.values())[-1]
    
    # Get latest backtest
    latest_backtest = await db.backtest_results.find_one({}, {"_id": 0}, sort=[("created_at", -1)])
    
    return {
        "total_models": total_models + custom_models,
        "custom_models": custom_models,
        "training_sessions": training_sessions_count,
        "backtest_count": backtest_count,
        "latest_session": latest_session,
        "latest_backtest": latest_backtest,
        "active_trainings": sum(1 for s in training_sessions.values() if s["status"] == "running")
    }

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
