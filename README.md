# Stock Price Prediction API com LSTM

Projeto desenvolvido como **Tech Challenge – Fase 4**, com o objetivo de criar
um modelo de **Deep Learning (LSTM)** para previsão de preços de ações e realizar
todo o pipeline de Machine Learning, incluindo **deploy em API**, **monitoramento**
e **containerização com Docker**.

---

## Tecnologias Utilizadas

- Python 3.12
- TensorFlow / Keras
- Scikit-learn
- FastAPI
- Uvicorn
- Docker
- Yahoo Finance (yfinance)

---

## Estrutura do Projeto

```text
tech-challenge-lstm/
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── data/
│   └── raw_data.csv
│
├── logs/
│   └── api.log
│
├── models/
│   ├── lstm_model.keras
│   └── scaler.pkl
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model_builder.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── inference.py
│
├── main.py
├── Dockerfile
├── requirements.txt
└── README.md