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
        self.best_models = self._load_best_models()
        
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
        
    def _load_best_models(self):
        try:
            metrics_dir = 'metrics/test'
            if not os.path.exists(metrics_dir):
                self.logger.error(f"Metrics directory not found: {metrics_dir}")
                return {}
                    
            timestamp_dirs = [d for d in os.listdir(metrics_dir) if os.path.isdir(os.path.join(metrics_dir, d))]
            if not timestamp_dirs:
                self.logger.error("No timestamp directories found in metrics/test")
                return {}
            
            latest_timestamp = sorted(timestamp_dirs)[-1]
            metrics_path = os.path.join(metrics_dir, latest_timestamp, 'all_metrics_summary.csv')
            
            if not os.path.exists(metrics_path):
                self.logger.error(f"Metrics file not found: {metrics_path}")
                return {}
            
            self.logger.info(f"Loading metrics from: {metrics_path}")
            metrics_df = pd.read_csv(metrics_path)
            best_models = {}
            
            for ticker in metrics_df['Ticker'].unique():
                ticker_metrics = metrics_df[metrics_df['Ticker'] == ticker]
                best_model = ticker_metrics.loc[ticker_metrics['F1'].idxmax()]
                best_models[ticker] = {
                    'model_name': best_model['Model'],
                    'f1_score': best_model['F1']
                }
                self.logger.info(f"Best model for {ticker}: {best_model['Model']} (F1: {best_model['F1']:.4f})")
            
            return best_models
        except Exception as e:
            self.logger.error(f"Error loading best models: {str(e)}")
            return {}

    def prepare_test_data(self, ticker, df):
        self.logger.info(f"Preparing test data for {ticker}...")
        
        data_prep = DataPreparation()
        ticker_df = df[df['Ticker'] == ticker].copy()
        
        if ticker_df.empty:
            self.logger.error(f"No data found for ticker: {ticker}")
            return None
        
        ticker_data = data_prep.prepare_features_by_ticker(ticker_df)
        
        if ticker not in ticker_data:
            self.logger.error(f"Failed to prepare features for ticker: {ticker}")
            return None
        
        split_data = data_prep.split_data_by_ticker(ticker_data)
        
        if ticker not in split_data:
            self.logger.error(f"Failed to split data for ticker: {ticker}")
            return None
        
        scaled_data = data_prep.scale_features_by_ticker({ticker: split_data[ticker]})
        
        if ticker not in scaled_data:
            self.logger.error(f"Failed to scale features for ticker: {ticker}")
            return None
        
        self.selected_features = data_prep.selected_features
        return scaled_data[ticker]

    def load_model(self, ticker, model_name=None):
        if model_name is None:
            if ticker not in self.best_models:
                self.logger.error(f"No best model found for ticker: {ticker}")
                return None
            model_name = self.best_models[ticker]['model_name']
            self.logger.info(f"Using best model for {ticker}: {model_name} (F1: {self.best_models[ticker]['f1_score']:.4f})")
        
        model_path = f'models/{ticker}/{model_name}_model.joblib'
        try:
            self.logger.info(f"Loading {model_name} model for {ticker}...")
            start_time = time.time()
            
            if model_name in ['lstm', 'transformer']:
                sample_data_path = f'data/sample/{ticker}_data.joblib'
                if os.path.exists(sample_data_path):
                    sample_data = joblib.load(sample_data_path)
                    input_shape = (sample_data['X_test_3d'].shape[1], sample_data['X_test_3d'].shape[2])
                    num_classes = len(np.unique(sample_data['y_test']))
                else:
                    input_shape = (WINDOW_SIZE, len(self.selected_features.get(ticker, [])))
                    num_classes = 2  # Binary classification
                
                model = self.models[model_name](
                    input_shape=input_shape,
                    num_classes=num_classes
                )
                model.model = joblib.load(model_path)
            else:
                model = self.models[model_name]()
                model.model = joblib.load(model_path)
            
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            model_size = sys.getsizeof(pickle.dumps(model.model)) / 1024 / 1024
            self.logger.info(f"Model size: {model_size:.2f} MB")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}", exc_info=True)
            return None

    def infer_last_point(self, ticker, df=None):
        """Get only the last data point for a ticker"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Getting last prediction point for ticker: {ticker}")
        self.logger.info(f"{'='*50}")
        
        if df is None:
            self.logger.info("Loading dataset...")
            df = DataPreparation().load_data(DATA_PATH)
        
        data = self.prepare_test_data(ticker, df)
        if data is None:
            return None
        
        if ticker not in self.best_models:
            self.logger.error(f"No best model found for ticker: {ticker}")
            return None
        
        model_name = self.best_models[ticker]['model_name']
        model = self.load_model(ticker)
        
        if model is None:
            self.logger.error(f"Failed to load model for {ticker}")
            return None
        
        X_test = data['X_test_3d'] if model_name in ['lstm', 'transformer'] else data['X_test_2d']
        test_dates = data['test_dates']
        y_test = data['y_test']
        
        try:
            self.logger.info(f"Making predictions for {ticker} using {model_name}")
            start_time = time.time()
            
            # Get only the last data point
            if len(X_test) > 0:
                last_X = X_test[-1:] 
                last_date = test_dates[-1]
                last_y_true = y_test[-1]
                
                y_pred = model.predict(last_X)
                
                try:
                    y_prob = model.predict_proba(last_X)
                except:
                    self.logger.warning(f"Probability predictions not available for {model_name}")
                    y_prob = None
                
                prediction_time = time.time() - start_time
                self.logger.info(f"Prediction completed in {prediction_time:.2f} seconds")
                
                # Prepare result for the last point only
                date_str = str(last_date)
                date_only = date_str.split(' ')[0] if ' ' in date_str else date_str
                
                result = {
                    'Ticker': ticker,
                    'Date': date_only,
                    'Prediction': int(y_pred[0]),
                    'model_used': model_name,
                    'f1_score': self.best_models[ticker]['f1_score']
                }
                
                if y_prob is not None:
                    result['confidence'] = float(y_prob[0][int(y_pred[0])])
                
                return result
            else:
                self.logger.warning(f"No test data available for {ticker}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return None

    def infer_all_last_points(self):
        """Get only the last data point for all tickers"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Getting last prediction point for all tickers")
        self.logger.info("="*50)
        
        df = DataPreparation().load_data(DATA_PATH)
        tickers = df['Ticker'].unique()
        
        self.logger.info(f"Found {len(tickers)} tickers: {', '.join(tickers)}")
        
        all_results = {}
        successful = []
        failed = []
        
        for ticker in tickers:
            try:
                self.logger.info(f"\nProcessing ticker: {ticker}")
                result = self.infer_last_point(ticker, df)
                
                if result:
                    all_results[ticker] = result
                    successful.append(ticker)
                    self.logger.info(f"Successfully got last prediction for {ticker}")
                else:
                    failed.append(ticker)
                    self.logger.warning(f"Failed to get prediction for {ticker}")
                    
            except Exception as e:
                failed.append(ticker)
                self.logger.error(f"Unexpected error for {ticker}: {str(e)}", exc_info=True)
        
        if all_results:
            self._save_all_last_points(all_results, successful)
        
        self._print_final_summary(tickers, successful, failed)
        
        return all_results

    def _save_all_last_points(self, all_results, successful):
        """Save the last data point for all tickers"""
        save_dir = os.path.join('inferences', self.timestamp)
        os.makedirs(save_dir, exist_ok=True)
        
        json_path = os.path.join(save_dir, 'all_last_points.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        all_rows = []
        for ticker, result in all_results.items():
            row = {
                'Ticker': ticker,
                'Date': result['Date'],
                'Prediction': result['Prediction'],
                'Model': result['model_used'],
                'F1_Score': result['f1_score']
            }
            if 'confidence' in result:
                row['Confidence'] = result['confidence']
            all_rows.append(row)
        
        if all_rows:
            df_results = pd.DataFrame(all_rows)
            csv_path = os.path.join(save_dir, 'all_last_points.csv')
            df_results.to_csv(csv_path, index=False)
            
            self.logger.info(f"\nSaved last points to {json_path} and {csv_path}")
            self.logger.info(f"Total predictions: {len(successful)}")
    
    def _print_final_summary(self, tickers, successful, failed):
        self.logger.info("\n" + "="*50)
        self.logger.info("FINAL INFERENCE SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total tickers: {len(tickers)}")
        self.logger.info(f"Successful predictions: {len(successful)}")
        self.logger.info(f"Failed predictions: {len(failed)}")
        
        if failed:
            self.logger.info("\nFailed tickers:")
            for ticker in failed:
                self.logger.info(f"- {ticker}")

def main():
    parser = argparse.ArgumentParser(description='Market Data Inference Tool')
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
        inferencer.logger.info(f"Getting last prediction for ticker: {args.ticker}")
        result = inferencer.infer_last_point(args.ticker, custom_df)
        if result:
            print(f"\nLast Prediction for {args.ticker}:")
            print(f"Date: {result['Date']}")
            print(f"Prediction: {result['Prediction']}")
            if 'confidence' in result:
                print(f"Confidence: {result['confidence']:.4f}")
            print(f"Model used: {result['model_used']}")
    elif args.all:
        inferencer.logger.info("Getting last prediction for all tickers")
        results = inferencer.infer_all_last_points()
    else:
        print("No action specified. Use --ticker or --all to get predictions.")
        parser.print_help()

if __name__ == "__main__":
    main()