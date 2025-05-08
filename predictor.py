import json
import numpy as np
import pandas as pd
from src.data_preparation import DataPreparation
from src.model import (SVMModel, LSTMModel, LogisticRegressionModel, TransformerModel)
from src.config import DATA_PATH, NUMBERS_FEATURES, WINDOW_SIZE
import os
import joblib
import logging
from datetime import datetime
import torch
import time
import psutil
import pickle
import sys
from src.evaluation import ModelEvaluation

class ModelPredictor:
    def __init__(self):
        self.models = {
            'svm': SVMModel,
            'logistic_regression': LogisticRegressionModel,
            'lstm': LSTMModel,
            'transformer': TransformerModel
        }
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._setup_logging()
        self.predictions = {}
        
    def _setup_logging(self):
        log_dir = 'logs/prediction'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'prediction_{timestamp}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info('=== Starting New Prediction Session ===')
    
    def load_model(self, ticker, model_name):
        model_path = f'models/{ticker}/{model_name}_model.joblib'
        try:
            self.logger.info(f"Loading {model_name} model for {ticker}...")
            start_time = time.time()
            
            if model_name in ['lstm', 'transformer']:
                # For deep learning models
                if model_name == 'lstm':
                    # Get number of features from the file structure
                    sample_data_path = f'data/sample/{ticker}_data.joblib'
                    if os.path.exists(sample_data_path):
                        sample_data = joblib.load(sample_data_path)
                        input_shape = (sample_data['X_test_3d'].shape[1], sample_data['X_test_3d'].shape[2])
                        num_classes = len(np.unique(sample_data['y_test']))
                    else:
                        # Default values if sample data not available
                        input_shape = (WINDOW_SIZE, NUMBERS_FEATURES)  # Default window size and features
                        num_classes = 2  # Binary classification
                    
                    model = self.models[model_name](
                        input_shape=input_shape,
                        num_classes=num_classes
                    )
                    model.model = joblib.load(model_path)
                    
                elif model_name == 'transformer':
                    # Similar approach for transformer
                    sample_data_path = f'data/sample/{ticker}_data.joblib'
                    if os.path.exists(sample_data_path):
                        sample_data = joblib.load(sample_data_path)
                        input_shape = (sample_data['X_test_3d'].shape[1], sample_data['X_test_3d'].shape[2])
                        num_classes = len(np.unique(sample_data['y_test']))
                    else:
                        input_shape = (WINDOW_SIZE, NUMBERS_FEATURES)  # Default window size and features
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
            
            if hasattr(model.model, 'n_features_in_'):
                self.logger.info(f"Number of features: {model.model.n_features_in_}")
                
            if model_name in ['lstm', 'transformer']:
                self.logger.info(f"Model architecture: {model.model}")
            
            return model
        
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {str(e)}", exc_info=True)
            return None

    def predict_test_data(self, model, X_test, ticker, model_name):
        try:
            if model is None:
                self.logger.warning(f"No model available for {ticker} - {model_name}")
                return None
                
            self.logger.info(f"Making predictions for {ticker} using {model_name}")
            self.logger.info(f"Test data shape: {X_test.shape}")
            
            start_time = time.time()
            y_pred = model.predict(X_test)
            
            try:
                y_prob = model.predict_proba(X_test)
                self.logger.info(f"Probability predictions shape: {y_prob.shape}")
            except:
                self.logger.warning(f"Probability predictions not available for {model_name}")
                y_prob = None
            
            prediction_time = time.time() - start_time
            self.logger.info(f"Predictions completed in {prediction_time:.2f} seconds")
            self.logger.info(f"Number of predictions: {len(y_pred)}")
            
            unique_preds, pred_counts = np.unique(y_pred, return_counts=True)
            pred_dist = dict(zip(map(str, unique_preds), map(int, pred_counts)))
            self.logger.info(f"Prediction distribution: {json.dumps(pred_dist, indent=2)}")
            
            return {
                'predictions': y_pred,
                'probabilities': y_prob
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting with {model_name} for {ticker}: {str(e)}")
            return None

    def save_predictions(self, predictions, ticker, data):
        try:
            save_dir = os.path.join('predictions', self.timestamp)
            os.makedirs(save_dir, exist_ok=True)
            self.logger.info(f"Created/verified predictions directory: {save_dir}")
            
            test_dates = data['test_dates']
            y_test = data['y_test']
            
            self.logger.info(f"Test dates length: {len(test_dates)}")
            self.logger.info(f"Test labels length: {len(y_test)}")
            
            results = []
            best_f1 = -1
            best_model = None
            evaluator = ModelEvaluation()
            
            for model_name, pred_data in predictions.items():
                if pred_data is not None:
                    y_pred = pred_data['predictions']
                    metrics = evaluator.evaluate(y_test, y_pred)
                    current_f1 = metrics['f1']
                    
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        best_model = model_name
            
            if best_model is not None:
                self.logger.info(f"Best model for {ticker}: {best_model} with F1 score {best_f1:.4f}")
                pred_data = predictions[best_model]
                for idx, pred in enumerate(pred_data['predictions']):
                    result = {
                        'Ticker': ticker,
                        'Model': best_model,
                        'Index': idx,
                        'Actual': y_test[idx],
                        'Prediction': pred,
                        'Month-Year': test_dates[idx]
                    }
                    
                    if pred_data['probabilities'] is not None:
                        probs = pred_data['probabilities'][idx]
                        for i, prob in enumerate(probs):
                            result[f'Prob_Class_{i}'] = prob
                    
                    results.append(result)
            
            if results:
                df = pd.DataFrame(results)
                self.logger.info(f"Created DataFrame with shape: {df.shape}")
                
                cols = ['Ticker', 'Model', 'Month-Year', 'Index', 'Actual', 'Prediction']
                prob_cols = [col for col in df.columns if col.startswith('Prob_Class_')]
                cols.extend(prob_cols)
                
                df = df[cols]
                
                df['Correct'] = (df['Actual'] == df['Prediction']).astype(int)
                
                # Calculate accuracy for best model
                accuracy = (df['Correct'].sum() / len(df)) * 100
                self.logger.info(f"Best model ({best_model}) - F1 Score: {best_f1:.4f}, Accuracy: {accuracy:.2f}%")
                
                save_path = os.path.join(save_dir, f'predictions_{ticker}.csv')
                self.logger.info(f"Attempting to save predictions for best model to: {save_path}")
                
                df.to_csv(save_path, index=False)
                
                self.logger.info(f"Successfully saved predictions to {save_path}")
                self.logger.info(f"File size: {os.path.getsize(save_path) / 1024:.2f} KB")
                self.logger.info(f"Columns in saved file: {', '.join(df.columns)}")
                return df
            else:
                self.logger.warning(f"No predictions to save for {ticker}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error saving predictions for {ticker}: {str(e)}")
            return None

    def evaluate_models(self, y_true, predictions, ticker):
        try:
            metrics_dir = os.path.join('metrics', 'test', self.timestamp)
            os.makedirs(metrics_dir, exist_ok=True)
            self.logger.info(f"Created/verified metrics directory: {metrics_dir}")
            
            results = []
            evaluator = ModelEvaluation()
            
            for model_name, pred_data in predictions.items():
                if pred_data is not None:
                    y_pred = pred_data['predictions']
                    metrics = evaluator.evaluate(y_true, y_pred)
                    
                    row = {
                        'Ticker': ticker,
                        'Model': model_name,
                        'Dataset': 'test',
                        'Accuracy': metrics.get('accuracy', float('nan')),
                        'Precision': metrics.get('precision', float('nan')),
                        'Recall': metrics.get('recall', float('nan')),
                        'F1': metrics.get('f1', float('nan')),
                    }
                    results.append(row)
                    
                    # Print classification report
                    self.logger.info(f"\n=== {model_name.upper()} Evaluation for {ticker} ===")
                    self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
                    self.logger.info(f"Precision: {metrics['precision']:.4f}")
                    self.logger.info(f"Recall: {metrics['recall']:.4f}")
                    self.logger.info(f"F1 Score: {metrics['f1']:.4f}")
                    
                    # Log confusion matrix or class distribution
                    unique_classes = sorted(set(y_true) | set(y_pred))
                    for cls in unique_classes:
                        actual_count = np.sum(y_true == cls)
                        pred_count = np.sum(y_pred == cls)
                        self.logger.info(f"Class {cls} - Actual: {actual_count}, Predicted: {pred_count}")
            
            if results:
                df = pd.DataFrame(results)
                save_path = os.path.join(metrics_dir, f'metrics_{ticker}.csv')
                df.to_csv(save_path, index=False)
                
                self.logger.info(f"Successfully saved metrics to {save_path}")
                
                self._append_to_summary_metrics(df, metrics_dir)
                
                return df
            else:
                self.logger.warning(f"No metrics to save for {ticker}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error evaluating models for {ticker}: {str(e)}")
            return None
    
    def _append_to_summary_metrics(self, df, metrics_dir):
        try:
            summary_path = os.path.join(metrics_dir, 'all_metrics_summary.csv')
            
            if os.path.exists(summary_path):
                existing_df = pd.read_csv(summary_path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_csv(summary_path, index=False)
            else:
                df.to_csv(summary_path, index=False)
            
            self.logger.info(f"Updated summary metrics file: {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Error updating summary metrics: {str(e)}")

def predict_single_ticker(ticker, df, predictor):
    predictor.logger.info(f"\n{'='*50}")
    predictor.logger.info(f"Starting prediction for ticker: {ticker}")
    predictor.logger.info(f"{'='*50}")
    
    # Prepare data
    data_prep = DataPreparation()
    ticker_df = df[df['Ticker'] == ticker].copy()
    
    if ticker_df.empty:
        predictor.logger.error(f"No data found for ticker: {ticker}")
        return False
    
    ticker_data = data_prep.prepare_features_by_ticker(ticker_df)
    
    if ticker not in ticker_data:
        predictor.logger.error(f"Failed to prepare features for ticker: {ticker}")
        return False
    
    split_data = data_prep.split_data_by_ticker(ticker_data)
    
    if ticker not in split_data:
        predictor.logger.error(f"Failed to split data for ticker: {ticker}")
        return False
    
    scaled_data = data_prep.scale_features_by_ticker({ticker: split_data[ticker]})
    
    if ticker not in scaled_data:
        predictor.logger.error(f"Failed to scale features for ticker: {ticker}")
        return False
    
    data = scaled_data[ticker]
    predictor.logger.info(f"Data prepared successfully for {ticker}")
    
    # Load all available models for this ticker
    ticker_predictions = {}
    model_dir = f'models/{ticker}'
    
    if not os.path.exists(model_dir):
        predictor.logger.error(f"No models directory found for ticker: {ticker}")
        return False
    
    predictor.logger.info(f"Looking for models in: {model_dir}")
    model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.joblib')]
    
    if not model_files:
        predictor.logger.error(f"No model files found for ticker: {ticker}")
        return False
    
    # Extract model names from filenames
    model_names = [f.split('_model.joblib')[0] for f in model_files]
    predictor.logger.info(f"Found models: {', '.join(model_names)}")
    
    # Make predictions with each model
    for model_name in model_names:
        # Skip if model_name is not in the supported models
        if model_name not in predictor.models:
            predictor.logger.warning(f"Model type {model_name} not supported. Skipping.")
            continue
            
        # Load the model
        model = predictor.load_model(ticker, model_name)
        
        if model is None:
            predictor.logger.warning(f"Failed to load {model_name} for {ticker}. Skipping.")
            continue
        
        # Select appropriate data format
        X_test = data['X_test_3d'] if model_name in ['lstm', 'transformer'] else data['X_test_2d']
        
        # Make predictions
        predictions = predictor.predict_test_data(model, X_test, ticker, model_name)
        if predictions is not None:
            ticker_predictions[model_name] = predictions
    
    # Evaluate and save results
    if ticker_predictions:
        predictor.evaluate_models(data['y_test'], ticker_predictions, ticker)
        predictor.save_predictions(ticker_predictions, ticker, data)
        predictor.logger.info(f"Completed prediction process for {ticker}")
        return True
    else:
        predictor.logger.warning(f"No successful predictions made for {ticker}")
        return False

def main():
    predictor = ModelPredictor()
    predictor.logger.info("Starting prediction process...")
    
    predictor.logger.info("Loading dataset...")
    df = DataPreparation().load_data(DATA_PATH)
    
    tickers = df['Ticker'].unique()
    predictor.logger.info(f"Found {len(tickers)} unique tickers: {', '.join(tickers)}")
    
    successful = []
    failed = []
    
    for ticker in tickers:
        try:
            predictor.logger.info(f"\nStarting prediction for ticker: {ticker}")
            result = predict_single_ticker(ticker, df, predictor)
            
            if result:
                successful.append(ticker)
                predictor.logger.info(f"Successfully completed predictions for {ticker}")
            else:
                failed.append(ticker)
                predictor.logger.warning(f"Failed prediction process for {ticker}")
                
        except Exception as e:
            failed.append(ticker)
            predictor.logger.error(f"Unexpected error for {ticker}: {str(e)}", exc_info=True)
    
    predictor.logger.info("\n" + "="*50)
    predictor.logger.info("PREDICTION SUMMARY")
    predictor.logger.info("="*50)
    predictor.logger.info(f"Total tickers: {len(tickers)}")
    predictor.logger.info(f"Successfully predicted: {len(successful)} tickers")
    predictor.logger.info(f"Failed: {len(failed)} tickers")
    
    if successful:
        predictor.logger.info(f"Successful tickers: {', '.join(successful)}")
    if failed:
        predictor.logger.info(f"Failed tickers: {', '.join(failed)}")
    
    # Generate summary of best models
    predictor.logger.info("\nGenerating best model summary...")
    try:
        pred_dir = os.path.join('predictions', predictor.timestamp)
        pred_files = [f for f in os.listdir(pred_dir) if f.startswith('predictions_') and f.endswith('.csv')]
        
        if pred_files:
            best_models = []
            for file in pred_files:
                ticker = file.replace('predictions_', '').replace('.csv', '')
                df = pd.read_csv(os.path.join(pred_dir, file))
                if not df.empty:
                    accuracy = (df['Correct'].sum() / len(df)) * 100
                    model = df['Model'].iloc[0]
                    best_models.append({
                        'Ticker': ticker,
                        'Best Model': model,
                        'Accuracy': accuracy
                    })
            
            if best_models:
                summary_df = pd.DataFrame(best_models)
                summary_path = os.path.join(pred_dir, 'best_models_summary.csv')
                summary_df.to_csv(summary_path, index=False)
                predictor.logger.info(f"Best model summary saved to {summary_path}")
                
                # Log the top performing models
                top_models = summary_df.sort_values('Accuracy', ascending=False).head(5)
                predictor.logger.info("\nTop 5 performing models:")
                for _, row in top_models.iterrows():
                    predictor.logger.info(f"{row['Ticker']}: {row['Best Model']} (Accuracy: {row['Accuracy']:.2f}%)")
        else:
            predictor.logger.warning("No prediction files found to generate summary")
            
    except Exception as e:
        predictor.logger.error(f"Error generating best model summary: {str(e)}")

if __name__ == "__main__":
    main()
