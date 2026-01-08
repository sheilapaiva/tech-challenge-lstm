import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ModelEvaluator:
    def __init__(self, scaler):
        self.scaler = scaler

    def evaluate(self, model, X_val, y_val):
        y_pred = model.predict(X_val)

        y_val_real = self.scaler.inverse_transform(y_val.reshape(-1, 1))
        y_pred_real = self.scaler.inverse_transform(y_pred)

        mae = mean_absolute_error(y_val_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_val_real, y_pred_real))
        mape = np.mean(
            np.abs((y_val_real - y_pred_real) / y_val_real)
        ) * 100

        return mae, rmse, mape
