import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import os
import joblib
import logging
from datetime import datetime
import time
import sys
import pickle
import uvicorn

from src.data_preparation import DataPreparation
from src.model import (SVMModel, LSTMModel, LogisticRegressionModel, TransformerModel)
from src.config import DATA_PATH, WINDOW_SIZE

app = FastAPI(
    title="Market Bubble Prediction API",
    description="API dự đoán bong bóng thị trường dựa trên dữ liệu chuỗi thời gian",
    version="1.0.0"
)

log_dir = 'logs/api'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f'api_{timestamp}.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model schemas
class PredictionRequest(BaseModel):
    ticker: str

class PredictionResponse(BaseModel):
    ticker: str
    date: str
    prediction: int
    prediction_label: str
    confidence: Optional[float] = None
    model_used: Optional[str] = None
    model_predictions: Dict[str, int]
    timestamp: str

class InferenceEngine:
    def __init__(self):
        self.models = {
            'svm': SVMModel,
            'logistic_regression': LogisticRegressionModel,
            'lstm': LSTMModel,
            'transformer': TransformerModel
        }
        self.selected_features = {}
        
    def load_model(self, ticker, model_name):
        model_path = f'models/{ticker}/{model_name}_model.joblib'
        try:
            logger.info(f"Loading {model_name} model for {ticker}...")
            
            if model_name in ['lstm', 'transformer']:
                # For deep learning models
                sample_data_path = f'data/sample/{ticker}_data.joblib'
                if os.path.exists(sample_data_path):
                    sample_data = joblib.load(sample_data_path)
                    input_shape = (sample_data['X_test_3d'].shape[1], sample_data['X_test_3d'].shape[2])
                    num_classes = len(np.unique(sample_data['y_test']))
                else:
                    # Default values if sample data not available
                    input_shape = (WINDOW_SIZE, len(self.selected_features.get(ticker, [])))
                    num_classes = 2  # Binary classification
                
                model = self.models[model_name](
                    input_shape=input_shape,
                    num_classes=num_classes
                )
                model.model = joblib.load(model_path)
            else:
                # For traditional ML models
                model = self.models[model_name]()
                model.model = joblib.load(model_path)
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            return None
    
    def prepare_last_window(self, ticker, df):
        logger.info(f"Preparing last data window for {ticker}...")
        
        data_prep = DataPreparation()
        ticker_df = df[df['Ticker'] == ticker].copy()
        
        if ticker_df.empty:
            logger.error(f"No data found for ticker: {ticker}")
            return None
        
        # Use the DataPreparation class to get selected features
        ticker_data = data_prep.prepare_features_by_ticker(ticker_df)
        
        if ticker not in ticker_data:
            logger.error(f"Failed to prepare features for ticker: {ticker}")
            return None
        
        # Save selected features for model loading
        self.selected_features = data_prep.selected_features
        
        # Get the features data
        X = ticker_data[ticker]['X']
        X = X.sort_values('year-month')
        
        # Get the last WINDOW_SIZE rows
        last_rows = X.tail(WINDOW_SIZE)
        current_date = last_rows.iloc[-1]['year-month']
        
        # Extract numerical features for the window
        selected_features = ['year-month'] + self.selected_features.get(ticker, [])
        last_rows = last_rows[selected_features]
        X_numeric = last_rows.select_dtypes(include=[np.number])
        
        # Create 2D (flattened) and 3D versions of the window
        X_2d = X_numeric.values.flatten().reshape(1, -1)
        X_3d = X_numeric.values.reshape(1, WINDOW_SIZE, -1)
        
        # Check for missing values, infinities
        X_2d = np.nan_to_num(X_2d, nan=0.0, posinf=0.0, neginf=0.0)
        X_3d = np.nan_to_num(X_3d, nan=0.0, posinf=0.0, neginf=0.0)
        
        return {
            'X_2d': X_2d,
            'X_3d': X_3d,
            'date': current_date
        }
    
    def infer_latest(self, ticker):
        """
        Make inference for the latest data point based on its window
        """
        logger.info(f"Starting inference for ticker: {ticker}")
        
        # Load dataset
        df = DataPreparation().load_data(DATA_PATH)
        
        # Prepare the last window of data
        window_data = self.prepare_last_window(ticker, df)
        
        if window_data is None:
            raise HTTPException(status_code=404, detail=f"Could not prepare data window for {ticker}")
        
        # Find available models for this ticker
        model_dir = f'models/{ticker}'
        if not os.path.exists(model_dir):
            raise HTTPException(status_code=404, detail=f"No models directory found for ticker: {ticker}")
        
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.joblib')]
        
        if not model_files:
            raise HTTPException(status_code=404, detail=f"No model files found for ticker: {ticker}")
        
        # Extract model names from filenames
        model_names = [f.split('_model.joblib')[0] for f in model_files]
        
        # Make predictions with each model
        predictions = {}
        model_confidences = {}
        
        for model_name in model_names:
            # Skip if model_name is not in the supported models
            if model_name not in self.models:
                continue
                
            # Load the model
            model = self.load_model(ticker, model_name)
            
            if model is None:
                continue
            
            # Select appropriate data format
            X = window_data['X_3d'] if model_name in ['lstm', 'transformer'] else window_data['X_2d']
            
            # Make prediction
            try:
                # Get prediction
                y_pred = model.predict(X)[0]
                
                # Try to get probability (confidence)
                try:
                    probs = model.predict_proba(X)[0]
                    confidence = float(probs[int(y_pred)])
                except:
                    confidence = None
                
                predictions[model_name] = int(y_pred)
                if confidence is not None:
                    model_confidences[model_name] = confidence
                
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {str(e)}")
        
        if not predictions:
            raise HTTPException(status_code=500, detail=f"No successful predictions made for {ticker}")
        
        # Determine final prediction (use highest confidence if available)
        best_model = None
        confidence = None
        
        if model_confidences:
            best_model = max(model_confidences, key=model_confidences.get)
            final_prediction = predictions[best_model]
            confidence = model_confidences[best_model]
        else:
            # Simple majority voting
            prediction_counts = {}
            for pred in predictions.values():
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
            final_prediction = max(prediction_counts, key=prediction_counts.get)
        
        # Format the response
        result = {
            'ticker': ticker,
            'date': str(window_data['date']),
            'prediction': int(final_prediction),
            'prediction_label': 'Bubble' if final_prediction == 1 else 'No Bubble',
            'model_predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        if best_model and confidence:
            result['model_used'] = best_model
            result['confidence'] = confidence
        
        return result

# Init engine
inference_engine = InferenceEngine()

@app.get("/")
def root():
    return {"message": "Market Bubble Prediction API - Use /predict/{ticker} to make predictions"}

@app.get("/predict/{ticker}", response_model=PredictionResponse)
def predict(ticker: str):
    """
    Dự đoán nhãn cho mã chứng khoán cụ thể dựa trên dữ liệu mới nhất
    """
    try:
        result = inference_engine.infer_latest(ticker)
        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error predicting for {ticker}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {str(e)}")

@app.get("/available-tickers")
def available_tickers():
    """
    Lấy danh sách các mã chứng khoán có sẵn
    """
    try:
        df = DataPreparation().load_data(DATA_PATH)
        tickers = df['Ticker'].unique().tolist()
        available_models = []
        
        for ticker in tickers:
            model_dir = f'models/{ticker}'
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.joblib')]
                if model_files:
                    available_models.append(ticker)
        
        return {
            "total": len(available_models),
            "tickers": available_models
        }
    except Exception as e:
        logger.error(f"Error getting available tickers: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving tickers: {str(e)}")

@app.get("/models/{ticker}")
def available_models(ticker: str):
    """
    Lấy danh sách các mô hình có sẵn cho một mã chứng khoán cụ thể
    """
    model_dir = f'models/{ticker}'
    if not os.path.exists(model_dir):
        raise HTTPException(status_code=404, detail=f"No models found for ticker: {ticker}")
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.joblib')]
    if not model_files:
        raise HTTPException(status_code=404, detail=f"No model files found for ticker: {ticker}")
    
    model_names = [f.split('_model.joblib')[0] for f in model_files]
    
    return {
        "ticker": ticker,
        "available_models": model_names
    }

@app.get("/predict-all")
def predict_all():
    """
    Dự đoán bong bóng thị trường cho tất cả các mã chứng khoán có sẵn
    """
    try:
        df = DataPreparation().load_data(DATA_PATH)
        tickers = df['Ticker'].unique().tolist()
        

        available_tickers = []
        for ticker in tickers:
            model_dir = f'models/{ticker}'
            if os.path.exists(model_dir) and any(f.endswith('_model.joblib') for f in os.listdir(model_dir)):
                available_tickers.append(ticker)
        
        if not available_tickers:
            raise HTTPException(status_code=404, detail="No tickers with available models found")
        
        logger.info(f"Making predictions for {len(available_tickers)} tickers")
        

        results = {}
        successful = []
        failed = []
        
        for ticker in available_tickers:
            try:
                result = inference_engine.infer_latest(ticker)
                if result:
                    results[ticker] = result
                    successful.append(ticker)
                    logger.info(f"Successfully completed inference for {ticker}")
                else:
                    failed.append(ticker)
                    logger.warning(f"Failed inference for {ticker}")
            except Exception as e:
                failed.append(ticker)
                logger.error(f"Error predicting for {ticker}: {str(e)}")

        summary = {
            "total_tickers": len(available_tickers),
            "successful_predictions": len(successful),
            "failed_predictions": len(failed),
            "timestamp": datetime.now().isoformat(),
            "predictions": results
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error performing predictions for all tickers: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing all predictions: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)