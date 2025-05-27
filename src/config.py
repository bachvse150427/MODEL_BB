import torch
import numpy as np

RANDOM_STATE = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_END_DATE = '2019-06-01'
VALID_END_DATE = '2022-06-01'

NOISE_SCALE = 0.025
# SMOTE_ALPHA_MIN = 0.5
# SMOTE_ALPHA_MAX = 1.0

WINDOW_SIZE = 4
MIN_SAMPLES_PER_CLASS = 6

LSTM_EPOCHS_OPTIMIZED = 111
LSTM_EPOCHS_TRAINED = 111
TRANSFORMER_EPOCHS_OPTIMIZED = 111
TRANSFORMER_EPOCHS_TRAINED = 111

SVM_TRIALS = 180
LR_TRIALS = 180
LSTM_TRIALS = 40
TRANSFORMER_TRIALS = 40

SVM_PARAM_RANGES = {
    'C': np.logspace(-6, -1, 6),
    'kernel': ['linear', 'rbf'],
    'gamma': np.logspace(-4, 0, 5),
    'class_weight': ['balanced'],
    'probability': [True],
    'tol': np.logspace(-3, -1, 3)
}

LR_PARAM_RANGES = {
    'C': np.logspace(-3, -1, 3),
    'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
    'max_iter': [2000, 3000, 5000],
    'class_weight': ['balanced'],
    'tol': np.logspace(-3, 1, 5)
}


LSTM_PARAM_RANGES = {
    'hidden_size': (8, 24),
    'num_layers': (1, 2),
    'dropout': (0.3, 0.7),
    'learning_rate': (0.0001, 0.001), 
    'batch_size': [4, 8],  
    'weight_decay': (1e-4, 1e-1) 
}

TRANSFORMER_PARAM_RANGES = {
    'd_model': (8, 24),  
    'num_layers': (1, 2),
    'dropout': (0.3, 0.7),
    'learning_rate': (0.0001, 0.001),
    'batch_size': [4, 8],
    'nhead': [2], 
    'weight_decay': (1e-4, 1e-1) 
}

FEATURE_COLUMNS = [

    'year-month',
    'ret', 'Volatility', 'HL', 'LO', 'PM',
    'MDD', 'TVV', 'SK', 'Median_HL', 'fore_di_rt',
    'variation_t', 'ma7_t', 'ma14_t', 'ma21_t', 's_d7_t',
    'RET', 'log_price',
]
# is_bubble


NUMBERS_FEATURES = 10

DATA_PATH = 'data/bubble.csv'
MODEL_PATH = 'models/model.joblib'
SCALER_PATH = 'models/scaler.joblib'
