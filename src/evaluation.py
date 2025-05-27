import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import os

class ModelEvaluation:
    @staticmethod
    def print_classification_report(y_true, y_pred):
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))
        
        print("\nClass Distribution:")
        unique_classes = sorted(set(y_true) | set(y_pred))
        
        pred_counts = pd.Series(y_pred).value_counts()
        true_counts = pd.Series(y_true).value_counts()
        
        print("\nPredicted class distribution:")
        for cls in unique_classes:
            count = pred_counts.get(cls, 0)
            percentage = (count / len(y_pred)) * 100
            print(f"Class {cls}: {count} samples ({percentage:.1f}%)")
        
        print("\nActual class distribution:")
        for cls in unique_classes:
            count = true_counts.get(cls, 0)
            percentage = (count / len(y_true)) * 100
            print(f"Class {cls}: {count} samples ({percentage:.1f}%)")
    
    @staticmethod
    def evaluate(y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
