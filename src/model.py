from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
from sklearn.linear_model import LogisticRegression
import math
import torch.nn.functional as F
from src.utils import EarlyStopping

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RANDOM_STATE, DEVICE

class BaseModel:
    def __init__(self):
        self.model = None
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names):
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(feature_names, self.model.feature_importances_))
        return None

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # Hiểu đúng thứ tự dữ liệu
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes) # =2 PHÂN LOẠI NHỊ PHÂN
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class LSTMModel(BaseModel):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_size = input_shape[1]
        self.hidden_size = 128
        self.num_layers = 2
        self.num_classes = num_classes
        self.device = torch.device(DEVICE)
        
        self.model = LSTMNet(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=num_classes,
            dropout=0.2
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def train(self, X_train, y_train, X_valid=None, y_valid=None, batch_size=32, epochs=100, logger=None):
        if len(X_train.shape) == 2:
            n_samples = X_train.shape[0]
            timesteps = self.input_shape[0]
            n_features = self.input_shape[1]
            X_train = X_train.reshape(n_samples, timesteps, n_features)
        
        X_tensor = torch.FloatTensor(X_train).to(DEVICE)
        y_tensor = torch.LongTensor(y_train).to(DEVICE)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, 
            batch_size=batch_size,
            shuffle=False 
        )
        
        if X_valid is not None and y_valid is not None:
            if len(X_valid.shape) == 2:
                n_samples = X_valid.shape[0]
                timesteps = self.input_shape[0]
                n_features = self.input_shape[1]
                X_valid = X_valid.reshape(n_samples, timesteps, n_features)
            
            X_valid_tensor = torch.FloatTensor(X_valid).to(DEVICE)
            y_valid_tensor = torch.LongTensor(y_valid).to(DEVICE)
            
            early_stopping = EarlyStopping(
                patience=10,
                verbose=True,
                logger=logger
            )
        
        log_message = f"Training LSTM with {epochs} epochs, batch size {batch_size}"
        if logger:
            logger.info(log_message)
        else:
            print(log_message)
            
        best_model_state = None
        best_valid_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if X_valid is not None and y_valid is not None:
                self.model.eval()
                
                with torch.no_grad():
                    valid_outputs = self.model(X_valid_tensor)
                    valid_loss = self.criterion(valid_outputs, y_valid_tensor).item()
                    
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_model_state = self.model.state_dict().copy()
                    
                    should_stop, best_state = early_stopping(valid_loss, self.model)
                    
                    if should_stop:
                        stop_message = f"Early stopping triggered at epoch {epoch+1}"
                        if logger:
                            logger.info(stop_message)
                        else:
                            print(stop_message)
                            
                        if best_state is not None:
                            self.model.load_state_dict(best_state)
                            best_model_state = best_state
                        break
                
                if (epoch + 1) % 10 == 0:
                    progress_message = f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Valid Loss: {valid_loss:.4f}"
                    if logger:
                        logger.info(progress_message)
                    else:
                        print(progress_message)
            else:
                if (epoch + 1) % 10 == 0:
                    progress_message = f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}"
                    if logger:
                        logger.info(progress_message)
                    else:
                        print(progress_message)
        
        if best_model_state is not None and X_valid is not None:
            self.model.load_state_dict(best_model_state)
            final_message = f"Training completed. Best validation loss: {best_valid_loss:.4f}"
            if logger:
                logger.info(final_message)
            else:
                print(final_message)
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()

class SVMModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            probability=True,
            class_weight='balanced',
            tol=1e-3,
            random_state=RANDOM_STATE
        )

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(
            C=0.1,
            solver='lbfgs',
            max_iter=2000,
            class_weight='balanced',
            tol=1e-4,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0
        )

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, 
                 dim_feedforward=256, dropout=0.1, max_seq_length=None):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        
        self.input_projection = nn.Linear(input_size, d_model)
        
        self.pos_encoder = LearnablePositionalEncoding(d_model, dropout, max_seq_length)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu"
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        out = encoded[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_seq_length=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_length, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        with torch.no_grad():
            self.position_embeddings.copy_(pe)
    
    def forward(self, x):
        x = x + self.position_embeddings[:, :x.size(1)]
        return self.dropout(x)

class TransformerModel(BaseModel):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.input_size = input_shape[1]
        self.seq_length = input_shape[0]
        self.num_classes = num_classes
        self.device = torch.device(DEVICE)
        
        self.d_model = 128
        self.nhead = 4      
        self.num_layers = 2
        self.dropout = 0.2
        
        self.model = TimeSeriesTransformer(
            input_size=self.input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            num_classes=num_classes,
            dim_feedforward=self.d_model * 2,
            dropout=self.dropout,
            max_seq_length=self.seq_length
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None, batch_size=32, epochs=100, logger=None):
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False 
        )
        
        if X_valid is not None and y_valid is not None:
            if len(X_valid.shape) == 2:
                n_samples = X_valid.shape[0]
                timesteps = self.seq_length
                n_features = self.input_size
                X_valid = X_valid.reshape(n_samples, timesteps, n_features)
            
            X_valid_tensor = torch.FloatTensor(X_valid).to(self.device)
            y_valid_tensor = torch.LongTensor(y_valid).to(self.device)
            
            early_stopping = EarlyStopping(patience=15, verbose=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            if X_valid is not None and y_valid is not None:
                self.model.eval()
                
                with torch.no_grad():
                    valid_outputs = self.model(X_valid_tensor)
                    valid_loss = self.criterion(valid_outputs, y_valid_tensor).item()
                    
                    should_stop, best_model = early_stopping(valid_loss, self.model)
                    if should_stop:
                        print(f"Early stopping at epoch {epoch+1}")
                        if best_model is not None:
                            self.model.load_state_dict(best_model)
                        break
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Valid Loss: {valid_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
