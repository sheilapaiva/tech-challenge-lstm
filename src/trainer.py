class ModelTrainer:
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train, X_val, y_val,
              epochs=20, batch_size=32):

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val)
        )

        return history
