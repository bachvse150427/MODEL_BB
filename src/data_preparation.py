import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier

from src.config import TRAIN_END_DATE, WINDOW_SIZE, MIN_SAMPLES_PER_CLASS, FEATURE_COLUMNS, RANDOM_STATE, VALID_END_DATE, NUMBERS_FEATURES, NOISE_SCALE

class DataPreparation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        self.selected_features = {}
        
    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        df['year-month'] = pd.to_datetime(df['year-month'])
        return df
    
    def select_best_features(self, ticker_data, n_features=5):
        self.logger.info(f"\nSelecting top {n_features} features for each ticker using Random Forest with data before {VALID_END_DATE}")
        
        for ticker, data in ticker_data.items():
            X = data['X']
            y = data['y']
            
            if isinstance(X.index, pd.DatetimeIndex):
                mask = X.index < pd.to_datetime(VALID_END_DATE)
            else:
                self.logger.warning(f"Index of data for ticker {ticker} is not DatetimeIndex. Attempting to convert...")
                try:
                    mask = pd.to_datetime(X.index) < pd.to_datetime(VALID_END_DATE)
                except Exception as e:
                    self.logger.error(f"Could not convert index to datetime for ticker {ticker}: {str(e)}")
                    continue
            
            X_before_valid = X[mask]
            y_before_valid = y[mask]
            
            self.logger.info(f"Ticker {ticker} - Using {len(X_before_valid)} samples before {VALID_END_DATE} for feature selection")
            
            numeric_features = X_before_valid.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_features) <= n_features:
                self.logger.info(f"Ticker {ticker} has {len(numeric_features)} numeric features, which is <= {n_features}. Using all features.")
                self.selected_features[ticker] = numeric_features
                continue
                
            try:
                rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
                X_numeric = X_before_valid[numeric_features]
                X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
                means = X_numeric.mean()
                X_numeric = X_numeric.fillna(means)
                
                for col in X_numeric.columns:
                    mean = X_numeric[col].mean()
                    std = X_numeric[col].std()
                    if std > 0:
                        upper_limit = mean + 5 * std
                        lower_limit = mean - 5 * std
                        X_numeric[col] = X_numeric[col].clip(lower_limit, upper_limit)
                
                y_valid = y_before_valid.dropna().astype(int)
                X_valid = X_numeric.loc[y_valid.index]
                X_array = X_valid.values
                
                if np.isnan(X_array).any() or np.isinf(X_array).any():
                    self.logger.warning(f"Ticker {ticker} still has NaN or inf values after cleaning. Replacing with zeros.")
                    X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
                    X_valid = pd.DataFrame(X_array, columns=X_valid.columns, index=X_valid.index)
                
                self.logger.info(f"Ticker {ticker} - Feature selection data shapes: X={X_valid.shape}, y={y_valid.shape}")
                unique_classes, class_counts = np.unique(y_valid, return_counts=True)
                self.logger.info(f"Class distribution before {VALID_END_DATE}: {dict(zip(unique_classes, class_counts))}")
                
                min_samples = min(class_counts)
                if min_samples < 2:
                    self.logger.warning(f"Ticker {ticker} has class(es) with less than 2 samples before {VALID_END_DATE}. Using all features.")
                    self.selected_features[ticker] = numeric_features
                    continue
                
                rf.fit(X_valid, y_valid)
                importances = rf.feature_importances_
                feature_importances = dict(zip(numeric_features, importances))
                sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
                top_features = [feat for feat, imp in sorted_features[:n_features]]
                
                self.logger.info(f"Ticker {ticker} - Selected features:")
                for feat, imp in sorted_features[:n_features]:
                    self.logger.info(f"  {feat}: {imp:.4f}")
                self.selected_features[ticker] = top_features
                
            except Exception as e:
                self.logger.error(f"Error selecting features for ticker {ticker}: {str(e)}")
                self.selected_features[ticker] = numeric_features
        
        return self.selected_features
    
    def prepare_features_by_ticker(self, df):
        df = df.sort_values(['Ticker', 'year-month'])
        ticker_data = {}
        
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].copy()
            X = ticker_df[FEATURE_COLUMNS].copy()
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                X.loc[:, col] = X[col].fillna(X[col].mean())
            y = pd.to_numeric(ticker_df['is_bubble'].shift(-1), errors='coerce')
            valid_idx = ~y.isna() | (y.index == y.index[-1])
            X = X.loc[valid_idx].copy()
            y = y.loc[valid_idx].fillna(0).astype(int)
            self.logger.info(f"After processing - X shape: {X.shape}, y shape: {y.shape}")
            self.logger.info(f"Last row target value: {y.iloc[-1]}")
            ticker_data[ticker] = {
                'X': X,
                'y': y
            }
        
        self.select_best_features(ticker_data, n_features=NUMBERS_FEATURES)
        
        return ticker_data
    
    def split_data_by_ticker(self, ticker_data, train_end_date=TRAIN_END_DATE, valid_end_date=VALID_END_DATE):
        train_end_date = pd.to_datetime(train_end_date)
        valid_end_date = pd.to_datetime(valid_end_date)
        split_data = {}
        
        for ticker, data in ticker_data.items():
            X = data['X']
            y = data['y']
            train_mask = X['year-month'] <= train_end_date
            valid_mask = (X['year-month'] > train_end_date) & (X['year-month'] <= valid_end_date)
            test_mask = X['year-month'] > valid_end_date
            X_train_full = X[train_mask]
            y_train_full = y[train_mask]
            X_valid_full = X[valid_mask]
            y_valid_full = y[valid_mask]
            X_test_full = X[test_mask]
            y_test_full = y[test_mask]
            selected_features = ['year-month'] + self.selected_features.get(ticker, [])
            self.logger.info(f"Ticker {ticker} - Using features: {selected_features}")
            X_train_full = X_train_full[selected_features]
            X_valid_full = X_valid_full[selected_features]
            X_test_full = X_test_full[selected_features]
            train_windows_2d = []
            train_windows_3d = []
            
            for i in range(len(X_train_full) - WINDOW_SIZE-1):
                X_window = X_train_full.iloc[i:i+WINDOW_SIZE]
                y_target = y_train_full.iloc[i+WINDOW_SIZE]
                X_numeric = X_window.select_dtypes(include=[np.number])
                train_windows_2d.append({
                    'X': X_numeric.values.flatten(),
                    'y': y_target
                })
                train_windows_3d.append({
                    'X': X_numeric.values,
                    'y': y_target
                })
            
            valid_windows_2d = []
            valid_windows_3d = []
            
            for i in range(len(X_valid_full) - WINDOW_SIZE-1):
                X_window = X_valid_full.iloc[i:i+WINDOW_SIZE]
                y_target = y_valid_full.iloc[i+WINDOW_SIZE]
                X_numeric = X_window.select_dtypes(include=[np.number])
                valid_windows_2d.append({
                    'X': X_numeric.values.flatten(),
                    'y': y_target,
                    'date': X_valid_full.iloc[i+WINDOW_SIZE]['year-month']
                })
                valid_windows_3d.append({
                    'X': X_numeric.values,
                    'y': y_target,
                    'date': X_valid_full.iloc[i+WINDOW_SIZE]['year-month']
                })
            
            test_windows_2d = []
            test_windows_3d = []
            
            for i in range(len(X_test_full) - WINDOW_SIZE):
                X_window = X_test_full.iloc[i:i+WINDOW_SIZE]
                y_target = y_test_full.iloc[i+WINDOW_SIZE]
                X_numeric = X_window.select_dtypes(include=[np.number])
                test_windows_2d.append({
                    'X': X_numeric.values.flatten(),
                    'y': y_target,
                    'date': X_test_full.iloc[i+WINDOW_SIZE]['year-month']
                })
                test_windows_3d.append({
                    'X': X_numeric.values,
                    'y': y_target,
                    'date': X_test_full.iloc[i+WINDOW_SIZE]['year-month']
                })
            
            if train_windows_2d and valid_windows_2d and test_windows_2d:
                X_train_2d = np.vstack([w['X'] for w in train_windows_2d])
                y_train = np.array([w['y'] for w in train_windows_2d])
                X_valid_2d = np.vstack([w['X'] for w in valid_windows_2d])
                y_valid = np.array([w['y'] for w in valid_windows_2d])
                X_test_2d = np.vstack([w['X'] for w in test_windows_2d])
                y_test = np.array([w['y'] for w in test_windows_2d])
                X_train_3d = np.array([w['X'] for w in train_windows_3d])
                X_valid_3d = np.array([w['X'] for w in valid_windows_3d])
                X_test_3d = np.array([w['X'] for w in test_windows_3d])
                valid_dates = [w['date'] for w in valid_windows_2d]
                test_dates = [w['date'] for w in test_windows_2d]
                self.logger.info(f"Ticker {ticker} - 2D shapes: X_train={X_train_2d.shape}, X_valid={X_valid_2d.shape}, X_test={X_test_2d.shape}")
                self.logger.info(f"Ticker {ticker} - 3D shapes: X_train={X_train_3d.shape}, X_valid={X_valid_3d.shape}, X_test={X_test_3d.shape}")
                split_data[ticker] = {
                    'X_train_2d': X_train_2d,
                    'X_valid_2d': X_valid_2d,
                    'X_test_2d': X_test_2d,
                    'X_train_3d': X_train_3d,
                    'X_valid_3d': X_valid_3d,
                    'X_test_3d': X_test_3d,
                    'y_train': y_train,
                    'y_valid': y_valid,
                    'y_test': y_test,
                    'valid_dates': valid_dates,
                    'test_dates': test_dates
                }
        
        return split_data
    
    def scale_features_by_ticker(self, split_data):
        scaled_data = {}
        
        for ticker, data in split_data.items():
            X_train_2d = data['X_train_2d']
            X_valid_2d = data['X_valid_2d']
            X_test_2d = data['X_test_2d']
            X_train_3d = data['X_train_3d']
            X_valid_3d = data['X_valid_3d']
            X_test_3d = data['X_test_3d']
            y_train = data['y_train']
            y_valid = data['y_valid']
            y_test = data['y_test']
            valid_dates = data['valid_dates']
            test_dates = data['test_dates']
            X_train_2d = np.nan_to_num(X_train_2d, nan=0.0, posinf=0.0, neginf=0.0)
            X_valid_2d = np.nan_to_num(X_valid_2d, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_2d = np.nan_to_num(X_test_2d, nan=0.0, posinf=0.0, neginf=0.0)
            X_train_3d = np.nan_to_num(X_train_3d, nan=0.0, posinf=0.0, neginf=0.0)
            X_valid_3d = np.nan_to_num(X_valid_3d, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_3d = np.nan_to_num(X_test_3d, nan=0.0, posinf=0.0, neginf=0.0)
            
            try:
                unique_classes, class_counts = np.unique(y_train, return_counts=True)
                if len(unique_classes) > 1 and all(count >= MIN_SAMPLES_PER_CLASS for count in class_counts):
                    self.logger.info(f"Applying TimeSeriesSMOTE for ticker {ticker}")
                    self.logger.info(f"Before SMOTE - X_3d shape: {X_train_3d.shape}, y shape: {y_train.shape}")
                    self.logger.info(f"Class distribution before SMOTE: {dict(zip(unique_classes, class_counts))}")
                    minority_class = unique_classes[np.argmin(class_counts)]
                    majority_class = unique_classes[np.argmax(class_counts)]
                    minority_indices = np.where(y_train == minority_class)[0]
                    majority_indices = np.where(y_train == majority_class)[0]
                    n_to_generate = len(majority_indices) - len(minority_indices)
                    if n_to_generate > 0 and n_to_generate < 5 * len(minority_indices):
                        self.logger.info(f"Generating {n_to_generate} synthetic minority samples")
                        synthetic_samples = []
                        synthetic_labels = []
                        np.random.seed(RANDOM_STATE)
                        for _ in range(n_to_generate):
                            idx1, idx2 = np.random.choice(minority_indices, 2, replace=len(minority_indices) < 2)
                            alpha = np.random.random()
                            new_sample = X_train_3d[idx1] * alpha + X_train_3d[idx2] * (1 - alpha)
                            noise = np.random.normal(0, NOISE_SCALE, new_sample.shape)
                            new_sample = new_sample + noise
                            synthetic_samples.append(new_sample)
                            synthetic_labels.append(minority_class)
                        X_train_3d = np.vstack([X_train_3d, np.array(synthetic_samples)])
                        y_train = np.append(y_train, synthetic_labels)
                        n_samples = X_train_3d.shape[0]
                        X_train_2d = X_train_3d.reshape(n_samples, -1)
                        unique_classes_after, class_counts_after = np.unique(y_train, return_counts=True)
                        self.logger.info(f"After TimeSeriesSMOTE - X_3d shape: {X_train_3d.shape}, y shape: {y_train.shape}")
                        self.logger.info(f"Class distribution after TimeSeriesSMOTE: {dict(zip(unique_classes_after, class_counts_after))}")
                    else:
                        self.logger.warning(f"Skipping TimeSeriesSMOTE - imbalance too severe or no need for resampling")
                else:
                    self.logger.warning(f"Skipping TimeSeriesSMOTE for ticker {ticker} - insufficient samples per class or only one class")
            except Exception as e:
                self.logger.error(f"Error applying TimeSeriesSMOTE for ticker {ticker}: {str(e)}")
                self.logger.error(f"Error details: {str(e.__class__.__name__)}: {str(e)}")
            
            scaler_2d = StandardScaler()
            X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
            X_valid_2d_scaled = scaler_2d.transform(X_valid_2d)
            X_test_2d_scaled = scaler_2d.transform(X_test_2d)
            n_samples_train, n_timesteps, n_features = X_train_3d.shape
            n_samples_valid = X_valid_3d.shape[0]
            n_samples_test = X_test_3d.shape[0]
            X_train_3d_reshaped = X_train_3d.reshape(n_samples_train, -1)
            X_valid_3d_reshaped = X_valid_3d.reshape(n_samples_valid, -1)
            X_test_3d_reshaped = X_test_3d.reshape(n_samples_test, -1)
            scaler_3d = StandardScaler()
            X_train_3d_scaled_flat = scaler_3d.fit_transform(X_train_3d_reshaped)
            X_valid_3d_scaled_flat = scaler_3d.transform(X_valid_3d_reshaped)
            X_test_3d_scaled_flat = scaler_3d.transform(X_test_3d_reshaped)
            X_train_3d_scaled = X_train_3d_scaled_flat.reshape(n_samples_train, n_timesteps, n_features)
            X_valid_3d_scaled = X_valid_3d_scaled_flat.reshape(n_samples_valid, n_timesteps, n_features)
            X_test_3d_scaled = X_test_3d_scaled_flat.reshape(n_samples_test, n_timesteps, n_features)
            self.logger.info(f"Ticker {ticker} - Final 3D shapes: X_train={X_train_3d_scaled.shape}, X_valid={X_valid_3d_scaled.shape}, X_test={X_test_3d_scaled.shape}")
            scaled_data[ticker] = {
                'X_train_2d': X_train_2d_scaled,
                'X_valid_2d': X_valid_2d_scaled,
                'X_test_2d': X_test_2d_scaled,
                'X_train_3d': X_train_3d_scaled,
                'X_valid_3d': X_valid_3d_scaled,
                'X_test_3d': X_test_3d_scaled,
                'y_train': y_train,
                'y_valid': y_valid,
                'y_test': y_test,
                'valid_dates': valid_dates,
                'test_dates': test_dates
            }
        
        return scaled_data
