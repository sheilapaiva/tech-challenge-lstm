from fastapi import FastAPI, HTTPException, Request
from api.schemas import PredictionRequest, PredictionResponse
from src.inference import StockPricePredictor

import logging
import time

# ============================
# CONFIGURAÇÃO DE LOGS
# ============================
logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ============================
# FASTAPI APP
# ============================
app = FastAPI(
    title="Stock Price Prediction API",
    description="API para previsão de preços de ações com LSTM",
    version="1.0.0"
)

MODEL_PATH = "models/lstm_model.keras"
SCALER_PATH = "models/scaler.pkl"
TIME_STEPS = 60

predictor = StockPricePredictor(MODEL_PATH, SCALER_PATH, TIME_STEPS)

# ============================
# MIDDLEWARE: TEMPO DE RESPOSTA
# ============================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logging.info(
        f"Endpoint={request.url.path} | "
        f"Method={request.method} | "
        f"Time={process_time:.4f}s | "
        f"Status={response.status_code}"
    )

    return response

# ============================
# HEALTH CHECK
# ============================
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}

# ============================
# PREVISÃO
# ============================
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        predicted_price = predictor.predict(request.prices)

        logging.info(
            f"Prediction generated | Price={predicted_price:.2f}"
        )

        return PredictionResponse(predicted_price=predicted_price)

    except ValueError as e:
        logging.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logging.error(f"Internal error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao gerar previsão"
        )
