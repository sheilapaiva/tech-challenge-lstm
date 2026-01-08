from pydantic import BaseModel
from typing import List


class PredictionRequest(BaseModel):
    prices: List[float]


class PredictionResponse(BaseModel):
    predicted_price: float
