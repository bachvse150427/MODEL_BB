import joblib
import os
import logging
from datetime import datetime

def setup_logging(log_dir='logs'):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(log_dir, f'training_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info('Starting new training session')
    return logging.getLogger()

def save_model(model, scaler, model_path, scaler_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logging.info(f'Model saved to {model_path}')
    logging.info(f'Scaler saved to {scaler_path}')

def load_model(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

class EarlyStopping:

    def __init__(self, patience=15, min_delta=0, verbose=True, logger=None, log_every=5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.verbose = verbose
        self.best_model_state = None
        self.logger = logger
        self.log_every = log_every
    
    def __call__(self, val_loss, model=None):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_model_state = model.state_dict().copy()
            if self.verbose:
                self._log_info(f"New best model found (loss: {val_loss:.6f})")
        else:
            self.counter += 1
            if self.verbose and self.log_every > 0 and (self.counter % self.log_every == 0 or self.counter == self.patience):
                self._log_info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self._log_info(f"Early stopping triggered after {self.counter} epochs")
        return self.early_stop, self.best_model_state
    
    def _log_info(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
