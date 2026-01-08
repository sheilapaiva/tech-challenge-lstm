from src.data_loader import StockDataLoader
from src.preprocessing import DataPreprocessor
from src.model_builder import LSTMModelBuilder
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.inference import StockPricePredictor

import joblib
import os


def main():
    # ==============================
    # CONFIGURAÇÕES GERAIS
    # ==============================
    SYMBOL = 'AAPL'
    START_DATE = '2018-01-01'
    END_DATE = '2024-07-20'
    TIME_STEPS = 60

    DATA_PATH = 'data/raw_data.csv'
    MODEL_PATH = 'models/lstm_model.keras'
    SCALER_PATH = 'models/scaler.pkl'

    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("🔹 Iniciando pipeline do Tech Challenge...\n")

    # ==============================
    # 1. COLETA DE DADOS
    # ==============================
    print("📥 Coletando dados...")
    loader = StockDataLoader(SYMBOL, START_DATE, END_DATE)
    df = loader.download_data()
    loader.save_to_csv(df, DATA_PATH)
    print(f"✅ Dados salvos em {DATA_PATH}\n")

    # ==============================
    # 2. PRÉ-PROCESSAMENTO
    # ==============================
    print("⚙️  Pré-processando dados...")
    preprocessor = DataPreprocessor(time_steps=TIME_STEPS)
    df_close = preprocessor.select_target(df)
    scaled_data = preprocessor.scale_data(df_close)
    X, y = preprocessor.create_sequences(scaled_data)
    X_train, X_val, y_train, y_val = preprocessor.split_data(X, y)
    print("✅ Pré-processamento concluído\n")

    # ==============================
    # 3. CONSTRUÇÃO DO MODELO
    # ==============================
    print("🧠 Construindo modelo LSTM...")
    model_builder = LSTMModelBuilder(TIME_STEPS)
    model = model_builder.build_model()
    print("✅ Modelo criado\n")

    # ==============================
    # 4. TREINAMENTO
    # ==============================
    print("🏋️ Treinando modelo...")
    trainer = ModelTrainer(model)
    trainer.train(X_train, y_train, X_val, y_val, epochs=20)
    print("✅ Treinamento finalizado\n")

    # ==============================
    # 5. AVALIAÇÃO
    # ==============================
    print("📊 Avaliando modelo...")
    evaluator = ModelEvaluator(preprocessor.scaler)
    mae, rmse, mape = evaluator.evaluate(model, X_val, y_val)

    print(f"📈 MAE: {mae:.2f}")
    print(f"📉 RMSE: {rmse:.2f}")
    print(f"📊 MAPE: {mape:.2f}%\n")

    # ==============================
    # 6. SALVAMENTO
    # ==============================
    print("💾 Salvando modelo e scaler...")
    model.save(MODEL_PATH)
    joblib.dump(preprocessor.scaler, SCALER_PATH)
    print("✅ Modelo e scaler salvos\n")

    # ==============================
    # 7. TESTE DE INFERÊNCIA
    # ==============================
    print("🔮 Testando inferência com dados reais...")
    predictor = StockPricePredictor(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        time_steps=TIME_STEPS
    )

    last_prices = df_close['Close'].values[-TIME_STEPS:].tolist()
    predicted_price = predictor.predict(last_prices)

    print(f"💰 Preço previsto para o próximo dia: ${predicted_price:.2f}")
    print("\n🎉 Pipeline executada com sucesso!")


if __name__ == "__main__":
    main()
