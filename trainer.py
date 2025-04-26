import json
import numpy as np
import pandas as pd
from src.data_preparation import DataPreparation
from src.model import (SVMModel, LSTMModel, LogisticRegressionModel, LSTMNet, TransformerModel, TimeSeriesTransformer)
from src.config import DATA_PATH, DEVICE, LSTM_EPOCHS_TRAINED, TRANSFORMER_EPOCHS_TRAINED
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
from src.utils import EarlyStopping

class ModelTrainer:
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
        log_dir = 'logs/training'
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info('=== Starting New Training Session ===')
        
    def load_best_params(self, ticker):
        params_path = f'models/{ticker}_best_params.json'
        try:
            with open(params_path, 'r') as f:
                params = json.load(f)
            self.logger.info(f"Successfully loaded parameters for {ticker}")
            return params
        except FileNotFoundError:
            self.logger.warning(f"No parameters file found for {ticker}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading parameters for {ticker}: {str(e)}")
            return None

    def train_and_evaluate_model(self, model_name, params, X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None, input_shape=None, num_classes=None):
        if params is None:
            self.logger.warning(f"Skipping {model_name} due to missing parameters")
            return None, None
            
        try:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Starting {model_name.upper()} Training")
            self.logger.info(f"Parameters: {json.dumps(params, indent=2)}")
            self.logger.info(f"Input Shape: {X_train.shape}")
            self.logger.info(f"Target Shape: {y_train.shape}")
            
            process = psutil.Process(os.getpid())
            self.logger.info(f"Memory Usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            class_dist = dict(zip(map(str, unique_classes), map(int, class_counts)))
            self.logger.info(f"Class Distribution: {json.dumps(class_dist, indent=2)}")
            
            start_time = time.time()
            
            if model_name == 'lstm':
                model = self.models[model_name](
                    input_shape=input_shape,
                    num_classes=num_classes
                )
                model.model = LSTMNet(
                    input_size=input_shape[1],
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers'],
                    num_classes=num_classes,
                    dropout=params['dropout']
                ).to(DEVICE)
                
                self.logger.info(f"LSTM Architecture:\n{str(model.model)}")
                self.logger.info(f"Device: {DEVICE}")
                self.logger.info(f"Batch Size: {params['batch_size']}")
                
                model.optimizer = torch.optim.Adam(
                    model.model.parameters(), 
                    lr=params['learning_rate'],
                    weight_decay=params.get('weight_decay', 0)
                )
                
                early_stopping = EarlyStopping(
                    patience=10, 
                    verbose=True,
                    logger=self.logger
                )
                
                X_train_reshaped = X_train
                
                if X_valid is not None:
                    X_valid_reshaped = X_valid
                
                X_tensor = torch.FloatTensor(X_train_reshaped).to(DEVICE)
                y_tensor = torch.LongTensor(y_train).to(DEVICE)
                dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
                dataloader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=params['batch_size'],
                    shuffle=False     
                )
                
                if X_valid is not None:
                    X_valid_tensor = torch.FloatTensor(X_valid_reshaped).to(DEVICE)
                    y_valid_tensor = torch.LongTensor(y_valid).to(DEVICE)
                
                validation_metrics = []
                
                for epoch in range(LSTM_EPOCHS_TRAINED):
                    model.model.train()
                    total_loss = 0
                    
                    for batch_X, batch_y in dataloader:
                        outputs = model.model(batch_X)
                        loss = model.criterion(outputs, batch_y)
                        
                        model.optimizer.zero_grad()
                        loss.backward()
                        model.optimizer.step()
                        
                        total_loss += loss.item()
                    
                    train_loss = total_loss / len(dataloader)
                    
                    valid_loss = None
                    valid_acc = None
                    if X_valid is not None:
                        model.model.eval()
                        with torch.no_grad():
                            valid_outputs = model.model(X_valid_tensor)
                            valid_loss = model.criterion(valid_outputs, y_valid_tensor).item()
                            
                            _, predicted = torch.max(valid_outputs.data, 1)
                            valid_acc = (predicted == y_valid_tensor).sum().item() / len(y_valid_tensor)
                            
                            validation_metrics.append({
                                'epoch': epoch + 1,
                                'loss': valid_loss,
                                'accuracy': valid_acc
                            })
                        
                        should_stop, best_state = early_stopping(valid_loss, model.model)
                        
                        if should_stop:
                            self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                            if best_state is not None:
                                model.model.load_state_dict(best_state)
                            break
                    
                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        valid_log = f", Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}" if valid_loss is not None else ""
                        self.logger.info(f"Epoch [{epoch+1}/{LSTM_EPOCHS_TRAINED}], Train Loss: {train_loss:.4f}{valid_log}")
                
                if validation_metrics:
                    best_epoch = min(range(len(validation_metrics)), key=lambda i: validation_metrics[i]['loss'])
                    best_val_loss = validation_metrics[best_epoch]['loss']
                    best_val_acc = validation_metrics[best_epoch]['accuracy']
                    
                    self.logger.info(f"Best validation metrics - Epoch: {validation_metrics[best_epoch]['epoch']}, "
                                     f"Loss: {best_val_loss:.4f}, Accuracy: {best_val_acc:.4f}")
            
            elif model_name == 'transformer':
                model = self.models[model_name](
                    input_shape=input_shape,
                    num_classes=num_classes
                )
                
                self.logger.info(f"Transformer input shape details: timesteps={input_shape[0]}, features={input_shape[1]}")
                
                d_model = params['d_model']
                nhead = params['nhead']
                
                d_model = (d_model // nhead) * nhead
                
                dim_feedforward = 2 * d_model
                
                model.model = TimeSeriesTransformer(
                    input_size=input_shape[1],
                    d_model=d_model,
                    nhead=nhead,
                    num_layers=params['num_layers'],
                    num_classes=num_classes,
                    dim_feedforward=dim_feedforward,
                    dropout=params['dropout'],
                    max_seq_length=input_shape[0]
                ).to(DEVICE)
                
                self.logger.info(f"Device: {DEVICE}")
                self.logger.info(f"Batch Size: {params['batch_size']}")
                
                model.optimizer = torch.optim.Adam(
                    model.model.parameters(),
                    lr=params['learning_rate'],
                    weight_decay=params.get('weight_decay', 0)
                )
                
                model.criterion = torch.nn.CrossEntropyLoss()
                
                early_stopping = EarlyStopping(
                    patience=10, 
                    verbose=True,
                    logger=self.logger
                )
                
                X_train_reshaped = X_train
                
                if X_valid is not None:
                    X_valid_reshaped = X_valid
                
                X_tensor = torch.FloatTensor(X_train_reshaped).to(DEVICE)
                y_tensor = torch.LongTensor(y_train).to(DEVICE)
                dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
                dataloader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=params['batch_size'],
                    shuffle=False 
                )
                
                if X_valid is not None:
                    X_valid_tensor = torch.FloatTensor(X_valid_reshaped).to(DEVICE)
                    y_valid_tensor = torch.LongTensor(y_valid).to(DEVICE)
                
                validation_metrics = []
                
                for epoch in range(TRANSFORMER_EPOCHS_TRAINED):
                    model.model.train()
                    total_loss = 0
                    
                    for batch_X, batch_y in dataloader:
                        outputs = model.model(batch_X)
                        loss = model.criterion(outputs, batch_y)
                        
                        model.optimizer.zero_grad()
                        loss.backward()
                        model.optimizer.step()
                        
                        total_loss += loss.item()
                    
                    train_loss = total_loss / len(dataloader)
                    
                    valid_loss = None
                    valid_acc = None
                    if X_valid is not None:
                        model.model.eval()
                        with torch.no_grad():
                            valid_outputs = model.model(X_valid_tensor)
                            valid_loss = model.criterion(valid_outputs, y_valid_tensor).item()
                            
                            _, predicted = torch.max(valid_outputs.data, 1)
                            valid_acc = (predicted == y_valid_tensor).sum().item() / len(y_valid_tensor)
                            
                            validation_metrics.append({
                                'epoch': epoch + 1,
                                'loss': valid_loss,
                                'accuracy': valid_acc
                            })
                        
                        should_stop, best_state = early_stopping(valid_loss, model.model)
                        
                        if should_stop:
                            self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                            if best_state is not None:
                                model.model.load_state_dict(best_state)
                            break
                    
                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        valid_log = f", Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}" if valid_loss is not None else ""
                        self.logger.info(f"Epoch [{epoch+1}/{TRANSFORMER_EPOCHS_TRAINED}], Train Loss: {train_loss:.4f}{valid_log}")
                
                if validation_metrics:
                    best_epoch = min(range(len(validation_metrics)), key=lambda i: validation_metrics[i]['loss'])
                    best_val_loss = validation_metrics[best_epoch]['loss']
                    best_val_acc = validation_metrics[best_epoch]['accuracy']
                    
                    self.logger.info(f"Best validation metrics - Epoch: {validation_metrics[best_epoch]['epoch']}, "
                                     f"Loss: {best_val_loss:.4f}, Accuracy: {best_val_acc:.4f}")
            
            else:
                model = self.models[model_name]()
                model.model.set_params(**params)
                model.train(X_train, y_train)
                
                if hasattr(model.model, 'n_features_in_'):
                    self.logger.info(f"Number of features used: {model.model.n_features_in_}")
                if hasattr(model.model, 'n_iter_'):
                    self.logger.info(f"Number of iterations: {model.model.n_iter_}")
            
            evaluator = ModelEvaluation()
            
            model_metrics = {}
            
            self.logger.info("\n=== Training Set Evaluation ===")
            y_train_pred = model.predict(X_train)
            
            train_metrics = evaluator.evaluate(y_train, y_train_pred)
            self.logger.info("\nTraining Metrics:")
            for metric, value in train_metrics.items():
                self.logger.info(f"{metric.capitalize()}: {value:.4f}")
            
            model_metrics['train'] = train_metrics
            
            self.logger.info("\nTraining Classification Report:")
            evaluator.print_classification_report(y_train, y_train_pred)
            
            if X_valid is not None and y_valid is not None:
                self.logger.info("\n=== Validation Set Evaluation ===")
                y_valid_pred = model.predict(X_valid)
                
                valid_metrics = evaluator.evaluate(y_valid, y_valid_pred)
                self.logger.info("\nValidation Metrics:")
                for metric, value in valid_metrics.items():
                    self.logger.info(f"{metric.capitalize()}: {value:.4f}")
                
                model_metrics['valid'] = valid_metrics
                
                self.logger.info("\nValidation Classification Report:")
                evaluator.print_classification_report(y_valid, y_valid_pred)
                
            if X_test is not None and y_test is not None:
                self.logger.info("\n=== Test Set Evaluation ===")
                y_test_pred = model.predict(X_test)
                
                test_metrics = evaluator.evaluate(y_test, y_test_pred)
                self.logger.info("\nTest Metrics:")
                for metric, value in test_metrics.items():
                    self.logger.info(f"{metric.capitalize()}: {value:.4f}")
                
                model_metrics['test'] = test_metrics
                
                self.logger.info("\nTest Classification Report:")
                evaluator.print_classification_report(y_test, y_test_pred)
                
            training_time = time.time() - start_time
            self.logger.info(f"\nTraining completed in {training_time:.2f} seconds")
            
            model_size = sys.getsizeof(pickle.dumps(model.model)) / 1024 / 1024
            self.logger.info(f"Model Size: {model_size:.2f} MB")
            
            self.logger.info(f"Successfully trained and evaluated {model_name}")
            self.logger.info(f"{'='*50}\n")
            return model, model_metrics
            
        except Exception as e:
            self.logger.error(f"Error training {model_name}: {str(e)}", exc_info=True)
            return None, None
        
    def save_model(self, model, ticker, model_name):
        if model is None:
            self.logger.warning(f"No model to save for {ticker} - {model_name}")
            return
            
        save_dir = f'models/{ticker}'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name}_model.joblib')
        
        try:
            start_time = time.time()
            joblib.dump(model.model, save_path)
            save_time = time.time() - start_time
            
            file_size = os.path.getsize(save_path) / 1024 / 1024
            
            self.logger.info(f"Model saved to {save_path}")
            self.logger.info(f"Save time: {save_time:.2f} seconds")
            self.logger.info(f"File size: {file_size:.2f} MB")
            
        except Exception as e:
            self.logger.error(f"Error saving model to {save_path}: {str(e)}", exc_info=True)

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
                
                # Tính accuracy cho model tốt nhất
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

    def save_metrics_by_ticker(self, ticker, metrics_data):
        try:
            metrics_dir = os.path.join('metrics', self.timestamp)
            os.makedirs(metrics_dir, exist_ok=True)
            self.logger.info(f"Created/verified metrics directory: {metrics_dir}")
            
            results = []
            
            for model_name, model_metrics in metrics_data.items():
                for dataset_type, metrics in model_metrics.items():
                    row = {
                        'Ticker': ticker,
                        'Model': model_name,
                        'Dataset': dataset_type,
                        'Accuracy': metrics.get('accuracy', float('nan')),
                        'Precision': metrics.get('precision', float('nan')),
                        'Recall': metrics.get('recall', float('nan')),
                        'F1': metrics.get('f1', float('nan')),
                    }
                    results.append(row)
            
            if results:
                df = pd.DataFrame(results)
                df = df.sort_values(by=['Model', 'Dataset'])
                
                save_path = os.path.join(metrics_dir, f'metrics_{ticker}.csv')
                df.to_csv(save_path, index=False)
                
                self.logger.info(f"Successfully saved metrics to {save_path}")
                self.logger.info(f"File size: {os.path.getsize(save_path) / 1024:.2f} KB")
                
                self._append_to_summary_metrics(df, metrics_dir)
                
                return df
            else:
                self.logger.warning(f"No metrics to save for {ticker}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error saving metrics for {ticker}: {str(e)}")
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

def train_single_ticker(ticker, df, trainer):
    trainer.logger.info(f"\n{'='*50}")
    trainer.logger.info(f"Starting training for ticker: {ticker}")
    trainer.logger.info(f"{'='*50}")
    
    data_prep = DataPreparation()
    ticker_df = df[df['Ticker'] == ticker].copy()
    
    if ticker_df.empty:
        trainer.logger.error(f"No data found for ticker: {ticker}")
        return False
    
    ticker_data = data_prep.prepare_features_by_ticker(ticker_df)
    
    if ticker not in ticker_data:
        trainer.logger.error(f"Failed to prepare features for ticker: {ticker}")
        return False
    
    split_data = data_prep.split_data_by_ticker(ticker_data)
    
    if ticker not in split_data:
        trainer.logger.error(f"Failed to split data for ticker: {ticker}")
        return False
    
    scaled_data = data_prep.scale_features_by_ticker({ticker: split_data[ticker]})
    
    if ticker not in scaled_data:
        trainer.logger.error(f"Failed to scale features for ticker: {ticker}")
        return False
    
    best_params = trainer.load_best_params(ticker)
    if best_params is None:
        trainer.logger.error(f"No optimization parameters found for {ticker}")
        return False
    
    trainer.logger.info(f"\nBest parameters from optimization:")
    for model_name, params in best_params.items():
        trainer.logger.info(f"{model_name.upper()}: {json.dumps(params, indent=2)}")
        
        optimization_log_path = os.path.join('logs/optimization', f'optimization_*.log')
        try:
            import glob
            log_files = glob.glob(optimization_log_path)
            if log_files:
                latest_log = max(log_files, key=os.path.getmtime)
                with open(latest_log, 'r') as f:
                    log_content = f.readlines()
                    
                validation_lines = [line for line in log_content 
                                   if ticker in line and model_name in line and "Validation" in line]
                
                if validation_lines:
                    for line in validation_lines[-3:]:
                        trainer.logger.info(f"Optimization validation: {line.strip()}")
        except Exception as e:
            trainer.logger.warning(f"Could not retrieve optimization validation metrics: {str(e)}")
    
    data = scaled_data[ticker]
    X_train_2d = data['X_train_2d']
    X_train_3d = data['X_train_3d']
    y_train = data['y_train']
    
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    min_class_count = min(class_counts)
    
    if min_class_count < 2:
        trainer.logger.warning(f"Ticker {ticker} has class(es) with less than 2 samples. Skipping training.")
        return False
    
    num_classes = len(np.unique(y_train))
    
    try:
        ticker_predictions = {}
        ticker_metrics = {}
        
        for model_name in ['svm', 'logistic_regression', 'lstm', 'transformer']:
            trainer.logger.info(f"\nTraining {model_name}...")
            
            X_train = X_train_3d if model_name in ['lstm', 'transformer'] else X_train_2d
            X_valid = data['X_valid_3d'] if model_name in ['lstm', 'transformer'] else data['X_valid_2d']
            X_test = data['X_test_3d'] if model_name in ['lstm', 'transformer'] else data['X_test_2d']
            y_valid = data['y_valid']
            y_test = data['y_test']
            
            model_params = best_params.get(model_name)
            
            if model_params is None:
                trainer.logger.warning(f"No optimized parameters found for {model_name}. Skipping.")
                continue
            
            model, model_metrics = trainer.train_and_evaluate_model(
                model_name=model_name,
                params=model_params,
                X_train=X_train,
                y_train=y_train,
                X_valid=X_valid, 
                y_valid=y_valid,
                X_test=X_test,
                y_test=y_test,
                input_shape=(X_train_3d.shape[1], X_train_3d.shape[2]) if model_name in ['lstm', 'transformer'] else None,
                num_classes=num_classes if model_name in ['lstm', 'transformer'] else None
            )
            
            if model_metrics is not None:
                ticker_metrics[model_name] = model_metrics
            
            predictions = trainer.predict_test_data(model, X_test, ticker, model_name)
            if predictions is not None:
                ticker_predictions[model_name] = predictions
            
            trainer.save_model(model, ticker, model_name)
        
        trainer.save_predictions(ticker_predictions, ticker, data)
        trainer.save_metrics_by_ticker(ticker, ticker_metrics)
        
        trainer.logger.info(f"\nAll models trained, predictions and metrics saved for {ticker}")
        return True
        
    except Exception as e:
        trainer.logger.error(f"Error in training process for {ticker}: {str(e)}", exc_info=True)
        return False

def main():
    trainer = ModelTrainer()
    trainer.logger.info("Starting sequential model training...")
    
    trainer.logger.info("Loading full dataset...")
    df = DataPreparation().load_data(DATA_PATH)
    
    tickers = df['Ticker'].unique()
    trainer.logger.info(f"Found {len(tickers)} unique tickers: {', '.join(tickers)}")
    
    successful = []
    failed = []
    
    for ticker in tickers:
        try:
            trainer.logger.info(f"\nStarting training for ticker: {ticker}")
            result = train_single_ticker(ticker, df, trainer)
            
            if result:
                successful.append(ticker)
                trainer.logger.info(f"Successfully trained models for {ticker}")
            else:
                failed.append(ticker)
                trainer.logger.warning(f"Skipped or failed training for {ticker}")
                
        except Exception as e:
            failed.append(ticker)
            trainer.logger.error(f"Unexpected error for {ticker}: {str(e)}", exc_info=True)
    
    trainer.logger.info("\n" + "="*50)
    trainer.logger.info("TRAINING SUMMARY")
    trainer.logger.info("="*50)
    trainer.logger.info(f"Total tickers: {len(tickers)}")
    trainer.logger.info(f"Successfully trained: {len(successful)} tickers")
    trainer.logger.info(f"Failed or skipped: {len(failed)} tickers")
    
    if successful:
        trainer.logger.info(f"Successful tickers: {', '.join(successful)}")
    if failed:
        trainer.logger.info(f"Failed tickers: {', '.join(failed)}")

if __name__ == "__main__":
    main()
