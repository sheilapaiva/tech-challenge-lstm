from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


class LSTMModelBuilder:
    def __init__(self, time_steps: int):
        self.time_steps = time_steps

    def build_model(self):
        model = Sequential()

        model.add(LSTM(
            units=50,
            return_sequences=True,
            input_shape=(self.time_steps, 1)
        ))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50))
        model.add(Dropout(0.2))

        model.add(Dense(1))

        model.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )

        return model
