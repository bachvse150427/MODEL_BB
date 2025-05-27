import json
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import torch
from src.model import SVMModel, LogisticRegressionModel, LSTMModel, LSTMNet, TransformerModel, TimeSeriesTransformer
from src.config import RANDOM_STATE, DEVICE, SVM_TRIALS, LR_TRIALS, LSTM_TRIALS, LSTM_EPOCHS_OPTIMIZED, TRANSFORMER_TRIALS, TRANSFORMER_EPOCHS_OPTIMIZED, LSTM_PARAM_RANGES, TRANSFORMER_PARAM_RANGES, SVM_PARAM_RANGES, LR_PARAM_RANGES
import optuna
import os
import logging
from datetime import datetime
import torch.nn as nn
from src.utils import EarlyStopping
from src.data_preparation import DataPreparation
from src.config import DATA_PATH

class ModelOptimizer:
    def __init__(self, X_train, y_train, X_valid, y_valid):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid 
        self.y_valid = y_valid 
        self.best_params = {}
        self.logger = None
        
    @classmethod
    def setup_logging(cls):
        if not hasattr(cls, 'logger'):
            log_dir = 'logs/optimization'
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = os.path.join(log_dir, f'optimization_{timestamp}.log')
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_filename),
                    logging.StreamHandler()
                ]
            )
            cls.logger = logging.getLogger(__name__)
            cls.logger.info('Starting hyperparameter optimization process')
        return cls.logger
  
    def optimize_svm(self, n_trials=SVM_TRIALS):
        self.logger.info("\nStarting SVM optimization")
        self.logger.info(f"Number of trials: {n_trials}")
        
        unique_classes = np.unique(self.y_train)
        if len(unique_classes) < 2:
            self.logger.warning(f"Only {len(unique_classes)} class found in training data. Skipping SVM optimization.")
            self.best_params['svm'] = None
            return None
        
        param_dist = SVM_PARAM_RANGES
        
        self.logger.info(f"Parameter distribution: {param_dist}")
        
        svm = SVMModel().model
        
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                
                random_search = RandomizedSearchCV(
                    svm, param_distributions=param_dist,
                    n_iter=n_trials, cv=None,
                    random_state=RANDOM_STATE,
                    n_jobs=-1, scoring='f1_weighted', error_score=0,
                    verbose=1
                )
                
                self.logger.info("Starting SVM Randomized Search CV...")
                random_search.fit(self.X_train, self.y_train)
            
            self.best_params['svm'] = random_search.best_params_
            best_model = random_search.best_estimator_
            valid_pred = best_model.predict(self.X_valid)
            from sklearn.metrics import f1_score, accuracy_score
            valid_f1 = f1_score(self.y_valid, valid_pred, average='weighted')
            valid_acc = accuracy_score(self.y_valid, valid_pred)
            self.logger.info(f"Best SVM parameters: {random_search.best_params_}")
            self.logger.info(f"Best score on train: {random_search.best_score_:.4f}")
            self.logger.info(f"Validation F1-score: {valid_f1:.4f}")
            self.logger.info(f"Validation Accuracy: {valid_acc:.4f}")
            return random_search.best_params_
        
        except Exception as e:
            self.logger.error(f"Error in SVM optimization: {str(e)}")
            self.best_params['svm'] = None
            return None
    
    def optimize_logistic_regression(self, n_trials=LR_TRIALS):
        self.logger.info("\nStarting Logistic Regression optimization")
        self.logger.info(f"Number of trials: {n_trials}")
        
        unique_classes = np.unique(self.y_train)
        if len(unique_classes) < 2:
            self.logger.warning(f"Only {len(unique_classes)} class found in training data. Skipping Logistic Regression optimization.")
            self.best_params['logistic_regression'] = None
            return None
        
        param_dist = LR_PARAM_RANGES.copy()
        param_dist['random_state'] = [RANDOM_STATE]
        
        self.logger.info(f"Parameter distribution: {param_dist}")
        
        lr = LogisticRegressionModel().model
        
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                random_search = RandomizedSearchCV(
                    lr, 
                    param_distributions=param_dist,
                    n_iter=n_trials, 
                    cv=None,
                    random_state=RANDOM_STATE,
                    n_jobs=-1, 
                    scoring='f1_weighted', 
                    error_score=0,
                    verbose=1
                )
                
                self.logger.info("Starting Logistic Regression Randomized Search CV...")
                random_search.fit(self.X_train, self.y_train)
            
            self.best_params['logistic_regression'] = random_search.best_params_
            best_model = random_search.best_estimator_
            valid_pred = best_model.predict(self.X_valid)
            from sklearn.metrics import f1_score, accuracy_score
            valid_f1 = f1_score(self.y_valid, valid_pred, average='weighted')
            valid_acc = accuracy_score(self.y_valid, valid_pred)
            self.logger.info(f"Best Logistic Regression parameters: {random_search.best_params_}")
            self.logger.info(f"Best score on train: {random_search.best_score_:.4f}")
            self.logger.info(f"Validation F1-score: {valid_f1:.4f}")
            self.logger.info(f"Validation Accuracy: {valid_acc:.4f}")
            return random_search.best_params_
        
        except Exception as e:
            self.logger.error(f"Error in Logistic Regression optimization: {str(e)}")
            self.best_params['logistic_regression'] = None
            return None

    def optimize_lstm(self, input_shape, num_classes, n_trials=LSTM_TRIALS):
        self.logger.info("\nStarting LSTM optimization")
        self.logger.info(f"Number of trials: {n_trials}")
        self.logger.info(f"Input shape: {input_shape}, Number of classes: {num_classes}")
        
        def objective(trial):
            try:
                params = {
                    'hidden_size': trial.suggest_int('hidden_size', 
                                                    LSTM_PARAM_RANGES['hidden_size'][0], 
                                                    LSTM_PARAM_RANGES['hidden_size'][1]),
                    'num_layers': trial.suggest_int('num_layers', 
                                                   LSTM_PARAM_RANGES['num_layers'][0], 
                                                   LSTM_PARAM_RANGES['num_layers'][1]),
                    'dropout': trial.suggest_float('dropout', 
                                                  LSTM_PARAM_RANGES['dropout'][0], 
                                                  LSTM_PARAM_RANGES['dropout'][1]),
                    'learning_rate': trial.suggest_float('learning_rate', 
                                                        LSTM_PARAM_RANGES['learning_rate'][0], 
                                                        LSTM_PARAM_RANGES['learning_rate'][1], 
                                                        log=True),
                    'batch_size': trial.suggest_categorical('batch_size', 
                                                           LSTM_PARAM_RANGES['batch_size']),
                    'weight_decay': trial.suggest_float('weight_decay', 
                                                        LSTM_PARAM_RANGES['weight_decay'][0], 
                                                        LSTM_PARAM_RANGES['weight_decay'][1],
                                                        log=True),
                    'fc_size': trial.suggest_int('fc_size',
                                             LSTM_PARAM_RANGES['fc_size'][0],
                                             LSTM_PARAM_RANGES['fc_size'][1]),
                    'epochs': LSTM_EPOCHS_OPTIMIZED
                }
                
                self.logger.info(f"Trial {trial.number} parameters: {params}")
                
                model = LSTMModel(
                    input_shape=input_shape,
                    num_classes=num_classes
                )
                
                model.model = LSTMNet(
                    input_size=input_shape[1],
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers'],
                    num_classes=num_classes,
                    dropout=params['dropout'],
                    fc_size=params['fc_size']
                ).to(DEVICE)
                
                model.optimizer = torch.optim.Adam(model.model.parameters(), lr=params['learning_rate'])
                
                X_train_reshaped = self.X_train
                
                X_tensor = torch.FloatTensor(X_train_reshaped).to(DEVICE)
                y_tensor = torch.LongTensor(self.y_train).to(DEVICE)
                dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
                dataloader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=params['batch_size'],
                    shuffle=False
                )
                
                early_stopping = EarlyStopping(
                    patience=10, 
                    verbose=True,
                    logger=self.logger
                )
                
                X_valid_tensor = torch.FloatTensor(self.X_valid).to(DEVICE)
                y_valid_tensor = torch.LongTensor(self.y_valid).to(DEVICE)
                
                try:
                    for epoch in range(params['epochs']):
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
                        
                        model.model.eval()
                        with torch.no_grad():
                            valid_outputs = model.model(X_valid_tensor)
                            valid_loss = model.criterion(valid_outputs, y_valid_tensor).item()
                        
                        model.model.train()
                        
                        should_stop, best_state = early_stopping(valid_loss, model.model)
                        
                        if should_stop:
                            if best_state is not None:
                                model.model.load_state_dict(best_state)
                            break
                        
                        if (epoch + 1) % 20 == 0:
                            self.logger.info(f"Epoch [{epoch+1}/{params['epochs']}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
                
                except Exception as e:
                    self.logger.error(f"Error during training: {str(e)}")
                    return float('inf')
                

                X_valid_reshaped = self.X_valid
                
                X_valid_tensor = torch.FloatTensor(X_valid_reshaped).to(DEVICE)
                y_valid_tensor = torch.LongTensor(self.y_valid).to(DEVICE)
                
                model.model.eval()
                
                with torch.no_grad():
                    valid_outputs = model.model(X_valid_tensor)
                    valid_loss = model.criterion(valid_outputs, y_valid_tensor).item()
                
                self.logger.info(f"Trial {trial.number} validation loss: {valid_loss:.4f}")
                return valid_loss
                
            except Exception as e:
                self.logger.error(f"Trial {trial.number} failed: {str(e)}")
                return float('inf')
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['lstm'] = study.best_params
        self.logger.info(f"Best LSTM parameters: {study.best_params}")
        self.logger.info(f"Best validation loss: {study.best_value:.4f}")

        model = LSTMModel(input_shape=input_shape, num_classes=num_classes)
        model.model = LSTMNet(
            input_size=input_shape[1],
            hidden_size=study.best_params['hidden_size'],
            num_layers=study.best_params['num_layers'],
            num_classes=num_classes,
            dropout=study.best_params['dropout']
        ).to(DEVICE)

        model.optimizer = torch.optim.Adam(
            model.model.parameters(), 
            lr=study.best_params['learning_rate'],
            weight_decay=study.best_params.get('weight_decay', 0)
        )

        X_train_tensor = torch.FloatTensor(self.X_train).to(DEVICE)
        y_train_tensor = torch.LongTensor(self.y_train).to(DEVICE)
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=study.best_params['batch_size'],
            shuffle=False
        )

        model.model.train()
        for epoch in range(min(20, LSTM_EPOCHS_OPTIMIZED)):
            for batch_X, batch_y in dataloader:
                outputs = model.model(batch_X)
                loss = model.criterion(outputs, batch_y)
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()

        model.model.eval()
        X_valid_tensor = torch.FloatTensor(self.X_valid).to(DEVICE)
        y_valid_tensor = torch.LongTensor(self.y_valid).to(DEVICE)
        with torch.no_grad():
            outputs = model.model(X_valid_tensor)
            _, predicted = torch.max(outputs.data, 1)
            valid_acc = (predicted == y_valid_tensor).sum().item() / len(y_valid_tensor)
            
        self.logger.info(f"Validation accuracy with best params: {valid_acc:.4f}")
        return study.best_params
    
    def optimize_transformer(self, input_shape, num_classes, n_trials=TRANSFORMER_TRIALS):
        self.logger.info("\nStarting Transformer optimization")
        self.logger.info(f"Number of trials: {n_trials}")
        self.logger.info(f"Input shape: {input_shape}, Number of classes: {num_classes}")
        
        def objective(trial):
            try:
                params = {
                    'd_model': trial.suggest_int('d_model', 
                                                TRANSFORMER_PARAM_RANGES['d_model'][0], 
                                                TRANSFORMER_PARAM_RANGES['d_model'][1]),
                    'num_layers': trial.suggest_int('num_layers', 
                                                   TRANSFORMER_PARAM_RANGES['num_layers'][0], 
                                                   TRANSFORMER_PARAM_RANGES['num_layers'][1]),
                    'dropout': trial.suggest_float('dropout', 
                                                  TRANSFORMER_PARAM_RANGES['dropout'][0], 
                                                  TRANSFORMER_PARAM_RANGES['dropout'][1]),
                    'learning_rate': trial.suggest_float('learning_rate', 
                                                        TRANSFORMER_PARAM_RANGES['learning_rate'][0], 
                                                        TRANSFORMER_PARAM_RANGES['learning_rate'][1], 
                                                        log=True),
                    'batch_size': trial.suggest_categorical('batch_size', 
                                                          TRANSFORMER_PARAM_RANGES['batch_size']),
                    'epochs': TRANSFORMER_EPOCHS_OPTIMIZED,
                    'nhead': trial.suggest_categorical('nhead', 
                                                      TRANSFORMER_PARAM_RANGES['nhead']),
                    'weight_decay': trial.suggest_float('weight_decay', 
                                                        TRANSFORMER_PARAM_RANGES['weight_decay'][0], 
                                                        TRANSFORMER_PARAM_RANGES['weight_decay'][1],
                                                        log=True),
                    'fc_size': trial.suggest_int('fc_size',
                                             TRANSFORMER_PARAM_RANGES['fc_size'][0],
                                             TRANSFORMER_PARAM_RANGES['fc_size'][1]),
                }
                
                nhead = params['nhead']
                d_model = params['d_model']
                adjusted_d_model = (d_model // nhead) * nhead
                
                if adjusted_d_model != d_model:
                    self.logger.info(f"d_model adjusted from {d_model} to {adjusted_d_model} to be divisible by nhead={nhead}")
                    
                params['d_model'] = adjusted_d_model
                
                self.logger.info(f"Trial {trial.number} parameters: {params}")
                
                model = TransformerModel(
                    input_shape=input_shape,
                    num_classes=num_classes
                )
                
                model.model = TimeSeriesTransformer(
                    input_size=input_shape[1],
                    d_model=params['d_model'],
                    nhead=params['nhead'],
                    num_layers=params['num_layers'],
                    num_classes=num_classes,
                    dim_feedforward=2 * params['d_model'],
                    dropout=params['dropout'],
                    fc_size=params['fc_size'],
                    max_seq_length=input_shape[0]
                ).to(DEVICE)
                
                model.optimizer = torch.optim.Adam(
                    model.model.parameters(),
                    lr=params['learning_rate']
                )
                
                model.criterion = nn.CrossEntropyLoss()
                
                X_train_reshaped = self.X_train
                
                X_tensor = torch.FloatTensor(X_train_reshaped).to(DEVICE)
                y_tensor = torch.LongTensor(self.y_train).to(DEVICE)
                dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
                dataloader = torch.utils.data.DataLoader(
                    dataset=dataset,
                    batch_size=params['batch_size'],
                    shuffle=False
                )
                
                X_valid_tensor = torch.FloatTensor(self.X_valid).to(DEVICE)
                y_valid_tensor = torch.LongTensor(self.y_valid).to(DEVICE)
                
                early_stopping = EarlyStopping(
                    patience=10, 
                    verbose=True,
                    logger=self.logger
                )

                try:
                    for epoch in range(params['epochs']):
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
                        
                        model.model.eval()
                        with torch.no_grad():
                            valid_outputs = model.model(X_valid_tensor)
                            valid_loss = model.criterion(valid_outputs, y_valid_tensor).item()
                        
                        model.model.train()
                        
                        should_stop, best_state = early_stopping(valid_loss, model.model)
                        
                        if should_stop:
                            if best_state is not None:
                                model.model.load_state_dict(best_state)
                            break
                        
                        if (epoch + 1) % 20 == 0:
                            self.logger.info(f"Epoch [{epoch+1}/{params['epochs']}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
                
                except Exception as e:
                    self.logger.error(f"Error during training: {str(e)}")
                    return float('inf')
                

                X_valid_reshaped = self.X_valid
                
                X_valid_tensor = torch.FloatTensor(X_valid_reshaped).to(DEVICE)
                y_valid_tensor = torch.LongTensor(self.y_valid).to(DEVICE)
                
                model.model.eval()
                
                with torch.no_grad():
                    valid_outputs = model.model(X_valid_tensor)
                    valid_loss = model.criterion(valid_outputs, y_valid_tensor).item()
                
                self.logger.info(f"Trial {trial.number} validation loss: {valid_loss:.4f}")
                return valid_loss
                
            except Exception as e:
                self.logger.error(f"Trial {trial.number} failed: {str(e)}")
                return float('inf')
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params['transformer'] = study.best_params
        self.logger.info(f"Best Transformer parameters: {study.best_params}")
        self.logger.info(f"Best validation loss: {study.best_value:.4f}")
        
        model = TransformerModel(input_shape=input_shape, num_classes=num_classes)
        
        nhead = study.best_params['nhead']
        d_model = study.best_params['d_model']
        adjusted_d_model = (d_model // nhead) * nhead
        
        model.model = TimeSeriesTransformer(
            input_size=input_shape[1],
            d_model=adjusted_d_model,
            nhead=nhead,
            num_layers=study.best_params['num_layers'],
            num_classes=num_classes,
            dim_feedforward=2 * adjusted_d_model,
            dropout=study.best_params['dropout'],
            max_seq_length=input_shape[0]
        ).to(DEVICE)
        
        model.optimizer = torch.optim.Adam(
            model.model.parameters(),
            lr=study.best_params['learning_rate'],
            weight_decay=study.best_params.get('weight_decay', 0)
        )
        
        model.criterion = nn.CrossEntropyLoss()
        
        X_train_tensor = torch.FloatTensor(self.X_train).to(DEVICE)
        y_train_tensor = torch.LongTensor(self.y_train).to(DEVICE)
        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=study.best_params['batch_size'],
            shuffle=False
        )
        
        model.model.train()
        for epoch in range(min(20, TRANSFORMER_EPOCHS_OPTIMIZED)):
            for batch_X, batch_y in dataloader:
                outputs = model.model(batch_X)
                loss = model.criterion(outputs, batch_y)
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()
        
        model.model.eval()
        X_valid_tensor = torch.FloatTensor(self.X_valid).to(DEVICE)
        y_valid_tensor = torch.LongTensor(self.y_valid).to(DEVICE)
        with torch.no_grad():
            outputs = model.model(X_valid_tensor)
            _, predicted = torch.max(outputs.data, 1)
            valid_acc = (predicted == y_valid_tensor).sum().item() / len(y_valid_tensor)
            
            from sklearn.metrics import f1_score
            predicted_np = predicted.cpu().numpy()
            y_valid_np = y_valid_tensor.cpu().numpy()
            valid_f1 = f1_score(y_valid_np, predicted_np, average='weighted')
        
        self.logger.info(f"Validation accuracy with best params: {valid_acc:.4f}")
        self.logger.info(f"Validation F1-score with best params: {valid_f1:.4f}")
        
        return study.best_params
    
    def save_best_params(self, output_path='models/best_params.json'):
        self.logger.info(f"Saving best parameters to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_numpy_types(obj.tolist())
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        output_data = convert_numpy_types(self.best_params)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        self.logger.info(f"Parameters saved to {output_path}")

def optimize_single_ticker(ticker, df, logger):
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting optimization for ticker: {ticker}")
    logger.info(f"{'='*50}")
    
    data_prep = DataPreparation()
    ticker_df = df[df['Ticker'] == ticker].copy()
    
    if ticker_df.empty:
        logger.error(f"No data found for ticker: {ticker}")
        return False
    
    ticker_data = data_prep.prepare_features_by_ticker(ticker_df)
    
    if ticker not in ticker_data:
        logger.error(f"Failed to prepare features for ticker: {ticker}")
        return False
    
    split_data = data_prep.split_data_by_ticker(ticker_data)
    
    if ticker not in split_data:
        logger.error(f"Failed to split data for ticker: {ticker}")
        return False
    
    scaled_data = data_prep.scale_features_by_ticker({ticker: split_data[ticker]})
    
    if ticker not in scaled_data:
        logger.error(f"Failed to scale features for ticker: {ticker}")
        return False
    
    data = scaled_data[ticker]
    X_train_2d = data['X_train_2d']
    X_valid_2d = data['X_valid_2d']
    X_train_3d = data['X_train_3d']
    X_valid_3d = data['X_valid_3d']
    y_train = data['y_train']
    y_valid = data['y_valid']
    
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    min_class_count = min(class_counts)
    
    if min_class_count < 2:
        logger.warning(f"Warning: Ticker {ticker} has class(es) with less than 2 samples. Skipping optimization.")
        return False
        
    try:
        num_classes = len(np.unique(y_train))
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Class distribution: {dict(zip(unique_classes, class_counts))}")
        
        optimizer = ModelOptimizer(X_train_2d, y_train, X_valid_2d, y_valid)
        optimizer.logger = logger
        
        logger.info("\nOptimizing SVM...")
        svm_params = optimizer.optimize_svm(n_trials=SVM_TRIALS)
        logger.info(f"Best SVM parameters: {svm_params}")
        
        logger.info("\nOptimizing Logistic Regression...")
        lr_params = optimizer.optimize_logistic_regression(n_trials=LR_TRIALS)
        logger.info(f"Best Logistic Regression parameters: {lr_params}")
        
        logger.info("\nOptimizing LSTM...")
        logger.info(f"X_train_3d shape: {X_train_3d.shape}")
        logger.info(f"X_train_3d dimensions: {len(X_train_3d.shape)}")

        if len(X_train_3d.shape) != 3:
            logger.warning(f"X_train_3d is not 3D! Reshaping...")
            X_train_3d = X_train_3d.reshape(X_train_3d.shape[0], 1, X_train_3d.shape[1])
            logger.info(f"After reshaping: {X_train_3d.shape}")

        _, timesteps, n_features = X_train_3d.shape
        input_shape = (timesteps, n_features)
        logger.info(f"LSTM input shape: {input_shape}")
        
        lstm_optimizer = ModelOptimizer(X_train_3d, y_train, X_valid_3d, y_valid)
        lstm_optimizer.logger = logger
        
        lstm_params = lstm_optimizer.optimize_lstm(
            input_shape=input_shape,
            num_classes=num_classes,
            n_trials=LSTM_TRIALS
        )
        logger.info(f"Best LSTM parameters: {lstm_params}")
        
        logger.info("\nOptimizing Transformer...")
        transformer_optimizer = ModelOptimizer(X_train_3d, y_train, X_valid_3d, y_valid)
        transformer_optimizer.logger = logger
        
        transformer_params = transformer_optimizer.optimize_transformer(
            input_shape=input_shape,
            num_classes=num_classes,
            n_trials=TRANSFORMER_TRIALS
        )
        logger.info(f"Best Transformer parameters: {transformer_params}")
        
        optimizer.best_params['lstm'] = lstm_params
        optimizer.best_params['transformer'] = transformer_params
        
        output_path = f'models/{ticker}_best_params.json'
        optimizer.save_best_params(output_path)
        logger.info(f"\nParameters saved to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error optimizing models for {ticker}: {str(e)}")
        logger.exception("Stack trace:")
        return False

def main():
    logger = ModelOptimizer.setup_logging()
    logger.info("Starting sequential model optimization...")

    logger.info("Loading full dataset...")
    df = DataPreparation().load_data(DATA_PATH)
    
    tickers = df['Ticker'].unique()
    logger.info(f"Found {len(tickers)} unique tickers: {', '.join(tickers)}")
    
    successful = []
    failed = []
    
    for ticker in tickers:
        try:
            logger.info(f"\nStarting optimization for ticker: {ticker}")
            result = optimize_single_ticker(ticker, df, logger)
            
            if result:
                successful.append(ticker)
                logger.info(f"Successfully optimized models for {ticker}")
            else:
                failed.append(ticker)
                logger.warning(f"Skipped or failed optimization for {ticker}")
                
        except Exception as e:
            failed.append(ticker)
            logger.error(f"Unexpected error for {ticker}: {str(e)}", exc_info=True)
    
    logger.info("\n" + "="*50)
    logger.info("OPTIMIZATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Total tickers: {len(tickers)}")
    logger.info(f"Successfully optimized: {len(successful)} tickers")
    logger.info(f"Failed or skipped: {len(failed)} tickers")
    
    if successful:
        logger.info(f"Successful tickers: {', '.join(successful)}")
    if failed:
        logger.info(f"Failed tickers: {', '.join(failed)}")


if __name__ == "__main__":
    main()
