import json
import numpy as np
import pandas as pd
from src.data_preparation import DataPreparation
from src.model import (SVMModel, LSTMModel, LogisticRegressionModel, TransformerModel)
from src.config import DATA_PATH, WINDOW_SIZE
import os
import joblib
import logging
from datetime import datetime
import torch
import time
import sys
import pickle
import argparse

class Inference:
    def __init__(self):
        self.models = {
            'svm': SVMModel,
            'logistic_regression': LogisticRegressionModel,
            'lstm': LSTMModel,
            'transformer': TransformerModel
        }
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._setup_logging()
        
    def _setup_logging(self):
        log_dir = 'logs/inference'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'inference_{timestamp}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info('=== Starting New Inference Session ===')
        
    def load_model(self, ticker, model_name):
        model_path = f'models/{ticker}/{model_name}_model.joblib'
        try:
            self.logger.info(f"Loading {model_name} model for {ticker}...")
            start_time = time.time()
            
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
            
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            # Log model details
            model_size = sys.getsizeof(pickle.dumps(model.model)) / 1024 / 1024
            self.logger.info(f"Model size: {model_size:.2f} MB")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}", exc_info=True)
            return None
    
    def prepare_last_window(self, ticker, df):
        self.logger.info(f"Preparing last data window for {ticker}...")
        
        data_prep = DataPreparation()
        ticker_df = df[df['Ticker'] == ticker].copy()
        
        if ticker_df.empty:
            self.logger.error(f"No data found for ticker: {ticker}")
            return None
        
        # Use the DataPreparation class to get selected features
        ticker_data = data_prep.prepare_features_by_ticker(ticker_df)
        
        if ticker not in ticker_data:
            self.logger.error(f"Failed to prepare features for ticker: {ticker}")
            return None
        
        # Save selected features for model loading
        self.selected_features = data_prep.selected_features
        
        # Get the features data
        X = ticker_data[ticker]['X']
        X = X.sort_values('year-month')
        
        # Get the last WINDOW_SIZE rows
        last_rows = X.tail(WINDOW_SIZE)
        current_date = last_rows.iloc[-1]['year-month']
        
        self.logger.info(f"Last data window ends at: {current_date}")
        self.logger.info(f"Window shape: {last_rows.shape}")
        
        # Extract numerical features for the window
        selected_features = ['year-month'] + self.selected_features.get(ticker, [])
        last_rows = last_rows[selected_features]
        X_numeric = last_rows.select_dtypes(include=[np.number])
        
        # Create 2D (flattened) and 3D versions of the window
        X_2d = X_numeric.values.flatten().reshape(1, -1)
        X_3d = X_numeric.values.reshape(1, WINDOW_SIZE, -1)
        
        # Apply scaling
        # Note: Ideally we should use the same scaler as during training
        # For simplicity, we're just ensuring the same data preparation steps
        
        # Check for missing values, infinities
        X_2d = np.nan_to_num(X_2d, nan=0.0, posinf=0.0, neginf=0.0)
        X_3d = np.nan_to_num(X_3d, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.logger.info(f"Prepared window shapes - 2D: {X_2d.shape}, 3D: {X_3d.shape}")
        
        return {
            'X_2d': X_2d,
            'X_3d': X_3d,
            'date': current_date
        }
    
    def infer_latest(self, ticker, df=None):
        """
        Make inference for the latest data point based on its window
        """
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Starting inference for ticker: {ticker}")
        self.logger.info(f"{'='*50}")
        
        if df is None:
            self.logger.info("Loading dataset...")
            df = DataPreparation().load_data(DATA_PATH)
        
        # Prepare the last window of data
        window_data = self.prepare_last_window(ticker, df)
        
        if window_data is None:
            self.logger.error(f"Could not prepare data window for {ticker}")
            return None
        
        # Find available models for this ticker
        model_dir = f'models/{ticker}'
        if not os.path.exists(model_dir):
            self.logger.error(f"No models directory found for ticker: {ticker}")
            return None
        
        self.logger.info(f"Looking for models in: {model_dir}")
        model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.joblib')]
        
        if not model_files:
            self.logger.error(f"No model files found for ticker: {ticker}")
            return None
        
        # Extract model names from filenames
        model_names = [f.split('_model.joblib')[0] for f in model_files]
        self.logger.info(f"Found models: {', '.join(model_names)}")
        
        # Make predictions with each model
        predictions = {}
        model_confidences = {}
        
        for model_name in model_names:
            # Skip if model_name is not in the supported models
            if model_name not in self.models:
                self.logger.warning(f"Model type {model_name} not supported. Skipping.")
                continue
                
            # Load the model
            model = self.load_model(ticker, model_name)
            
            if model is None:
                self.logger.warning(f"Failed to load {model_name} for {ticker}. Skipping.")
                continue
            
            # Select appropriate data format
            X = window_data['X_3d'] if model_name in ['lstm', 'transformer'] else window_data['X_2d']
            
            # Make prediction
            try:
                start_time = time.time()
                
                # Get prediction
                y_pred = model.predict(X)[0]
                
                # Try to get probability (confidence)
                try:
                    probs = model.predict_proba(X)[0]
                    confidence = float(probs[int(y_pred)])
                except:
                    confidence = None
                
                pred_time = time.time() - start_time
                
                predictions[model_name] = int(y_pred)
                if confidence is not None:
                    model_confidences[model_name] = confidence
                
                self.logger.info(f"Model {model_name} predicts class {y_pred} for {ticker} " + 
                              (f"with {confidence:.2%} confidence " if confidence else "") + 
                              f"(in {pred_time:.4f} seconds)")
                
            except Exception as e:
                self.logger.error(f"Error predicting with {model_name}: {str(e)}")
        
        if not predictions:
            self.logger.error(f"No successful predictions made for {ticker}")
            return None
        
        # Determine final prediction (use highest confidence if available)
        if model_confidences:
            best_model = max(model_confidences, key=model_confidences.get)
            final_prediction = predictions[best_model]
            confidence = model_confidences[best_model]
            self.logger.info(f"Selected model {best_model} with highest confidence: {confidence:.2%}")
        else:
            # Simple majority voting
            prediction_counts = {}
            for pred in predictions.values():
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
            final_prediction = max(prediction_counts, key=prediction_counts.get)
            vote_count = prediction_counts[final_prediction]
            total_votes = len(predictions)
            self.logger.info(f"Selected prediction {final_prediction} by majority vote: {vote_count}/{total_votes} models")
        
        # Save the inference result
        result = {
            'ticker': ticker,
            'date': window_data['date'],
            'prediction': int(final_prediction),
            'model_predictions': predictions
        }
        
        if model_confidences:
            result['model_confidences'] = {model: float(conf) for model, conf in model_confidences.items()}
        
        self._save_inference_result(result)
        
        return result
    
    def _save_inference_result(self, result):
        save_dir = os.path.join('inferences', self.timestamp)
        os.makedirs(save_dir, exist_ok=True)
        
        ticker = result['ticker']
        save_path = os.path.join(save_dir, f'inference_{ticker}.json')
        
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        self.logger.info(f"Saved inference result to {save_path}")
        
        # Also save as CSV for easier analysis
        csv_path = os.path.join(save_dir, f'inference_{ticker}.csv')
        
        df = pd.DataFrame([{
            'Ticker': result['ticker'],
            'Date': result['date'],
            'Prediction': result['prediction'],
            'Probability': result.get('model_confidences', {}).get(max(result.get('model_confidences', {}), key=result.get('model_confidences', {}).get, default=None), None)
        }])
        
        df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved inference result to CSV: {csv_path}")
        
        return save_path
    
    def infer_all_tickers(self):
        """
        Make inferences for all available tickers in the dataset
        """
        self.logger.info("Starting inference for all tickers...")
        
        df = DataPreparation().load_data(DATA_PATH)
        tickers = df['Ticker'].unique()
        
        self.logger.info(f"Found {len(tickers)} tickers: {', '.join(tickers)}")
        
        results = {}
        successful = []
        failed = []
        
        for ticker in tickers:
            try:
                result = self.infer_latest(ticker, df)
                
                if result:
                    results[ticker] = result
                    successful.append(ticker)
                    self.logger.info(f"Successfully completed inference for {ticker}")
                else:
                    failed.append(ticker)
                    self.logger.warning(f"Failed inference for {ticker}")
                    
            except Exception as e:
                failed.append(ticker)
                self.logger.error(f"Unexpected error for {ticker}: {str(e)}", exc_info=True)
        
        self.logger.info("\n" + "="*50)
        self.logger.info("INFERENCE SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total tickers: {len(tickers)}")
        self.logger.info(f"Successful inferences: {len(successful)}")
        self.logger.info(f"Failed inferences: {len(failed)}")
        
        # Save all results to a single file
        if results:
            combined_save_dir = os.path.join('inferences', self.timestamp)
            combined_path = os.path.join(combined_save_dir, 'all_inferences.json')
            
            with open(combined_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Saved combined inference results to {combined_path}")
            
            # Also create a CSV summary
            summary_rows = []
            for ticker, result in results.items():
                row = {
                    'Ticker': ticker,
                    'Date': result['date'],
                    'Prediction': result['prediction']
                }
                summary_rows.append(row)
            
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                csv_path = os.path.join(combined_save_dir, 'all_inferences.csv')
                summary_df.to_csv(csv_path, index=False)
                self.logger.info(f"Saved combined inference results as CSV to {csv_path}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description=' Market Inference Tool')
    parser.add_argument('--ticker', type=str, help='Specific ticker to make inference for')
    parser.add_argument('--all', action='store_true', help='Make inferences for all tickers')
    parser.add_argument('--csv', type=str, help='Path to a specific CSV file to use')
    
    args = parser.parse_args()
    
    inferencer = Inference()
    
    if args.csv:
        custom_df = pd.read_csv(args.csv)
        custom_df['year-month'] = pd.to_datetime(custom_df['year-month'])
        inferencer.logger.info(f"Using custom CSV file: {args.csv}")
    else:
        custom_df = None
    
    if args.ticker:
        inferencer.logger.info(f"Making inference for specific ticker: {args.ticker}")
        result = inferencer.infer_latest(args.ticker, custom_df)
        
        if result:
            print(f"\nInference for {args.ticker}:")
            print(f"Date: {result['date']}")
            print(f"Prediction: {'1' if result['prediction'] == 1 else 'No 1'} (class {result['prediction']})")
            
            if 'model_confidences' in result:
                best_model = max(result['model_confidences'], key=result['model_confidences'].get)
                confidence = result['model_confidences'][best_model]
                print(f"Confidence: {confidence:.2%} (using {best_model} model)")
        else:
            print(f"Failed to make inference for {args.ticker}")
    
    elif args.all:
        inferencer.logger.info("Making inferences for all available tickers")
        results = inferencer.infer_all_tickers()
        
        if results:
            print("\nInference Results Summary:")
            for ticker, result in results.items():
                print(f"{ticker}: {'1' if result['prediction'] == 1 else 'No 1'} " + 
                      f"(class {result['prediction']}) on {result['date']}")
    
    else:
        print("No action specified. Use --ticker or --all to make inferences.")
        parser.print_help()

if __name__ == "__main__":
    main()