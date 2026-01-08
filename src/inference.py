import numpy as np
import joblib
from tensorflow.keras.models import load_model


class StockPricePredictor:
    def __init__(self, model_path: str, scaler_path: str, time_steps: int):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        self.time_steps = time_steps

    def prepare_input(self, prices: list):
        if len(prices) < self.time_steps:
            raise ValueError(f"São necessários pelo menos {self.time_steps} valores")

        prices = np.array(prices).reshape(-1, 1)
        prices_scaled = self.scaler.transform(prices)

        X = prices_scaled[-self.time_steps:]
        return X.reshape((1, self.time_steps, 1))

    def predict(self, prices: list):
        X = self.prepare_input(prices)
        prediction_scaled = self.model.predict(X)
        prediction = self.scaler.inverse_transform(prediction_scaled)
        return float(prediction[0][0])
