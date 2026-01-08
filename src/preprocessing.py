import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    def __init__(self, time_steps: int = 60):
        self.time_steps = time_steps
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def select_target(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[['Close']]

    def scale_data(self, data: pd.DataFrame) -> np.ndarray:
        return self.scaler.fit_transform(data)

    def create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(self.time_steps, len(data)):
            X.append(data[i - self.time_steps:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def split_data(self, X, y, train_ratio=0.8):
        split = int(len(X) * train_ratio)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

        return X_train, X_val, y_train, y_val
