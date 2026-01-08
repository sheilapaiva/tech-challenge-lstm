# 📈 Stock Price Prediction API com LSTM

Projeto desenvolvido como **Tech Challenge – Fase 4**, com o objetivo de criar
um modelo de **Deep Learning (LSTM)** para previsão de preços de ações e realizar
todo o pipeline de Machine Learning, incluindo **deploy em API**, **monitoramento**
e **containerização com Docker**.

## 📌 Ativo Financeiro Utilizado

Neste projeto, o modelo foi treinado utilizando dados históricos da ação
**Apple Inc. (AAPL)**, negociada na bolsa de valores norte-americana (NASDAQ).

A escolha da AAPL se deve a:
- Alta liquidez
- Grande volume de negociações
- Série histórica consistente
- Ampla disponibilidade de dados públicos

Embora o modelo tenha sido treinado com dados da AAPL, a arquitetura e o pipeline
foram desenvolvidos de forma genérica, permitindo fácil adaptação para outros
ativos financeiros, bastando alterar o ticker e realizar novo treinamento.

---

## 🚀 Tecnologias Utilizadas

- Python 3.12
- TensorFlow / Keras
- Scikit-learn
- FastAPI
- Uvicorn
- Docker
- Yahoo Finance (yfinance)

---

## 📁 Estrutura do Projeto

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
```

## 🧠 Descrição do Modelo

O modelo utiliza uma arquitetura LSTM (Long Short-Term Memory) para capturar
padrões temporais nos preços históricos de fechamento de ações.
- Entrada: últimos 60 preços de fechamento
- Saída: previsão do próximo preço
- Métricas utilizadas:

  - MAE
  - RMSE
  - MAPE

## 🏋️ Detalhes do Treinamento

O modelo foi treinado utilizando uma abordagem de aprendizado supervisionado,
com base em séries temporais.

### Configurações principais:
- Tamanho da janela temporal (time steps): 60 dias
- Função de perda: Mean Squared Error (MSE)
- Otimizador: Adam
- Número de épocas: 20
- Batch size: 32
- Divisão dos dados:
  - 80% para treino
  - 20% para validação

Antes do treinamento, os dados foram normalizados utilizando o
**MinMaxScaler**, garantindo que os valores estivessem no intervalo [0, 1],
o que é essencial para o bom desempenho de redes neurais LSTM.
 
## ⚙️ Pipeline de Machine Learning

1. Coleta de dados com Yahoo Finance

2. Pré-processamento e normalização

3. Criação de janelas temporais

4. Treinamento do modelo LSTM

5. Avaliação com métricas

6. Salvamento do modelo e scaler

7. Deploy via API REST

8. Monitoramento e logs

## 📊 Fonte e Características dos Dados

Os dados utilizados no treinamento do modelo foram obtidos por meio da biblioteca
**yfinance**, que fornece acesso a dados financeiros históricos do Yahoo Finance.

### Características do dataset:
- Ativo: AAPL (Apple Inc.)
- Período: Janeiro de 2018 até Julho de 2024
- Frequência: Diária
- Variáveis disponíveis:
  - Open
  - High
  - Low
  - Close
  - Adj Close
  - Volume

Para o desenvolvimento do modelo, foi utilizada exclusivamente a variável
**Close (preço de fechamento)**, por ser amplamente empregada em análises
financeiras e representar o valor final de negociação do ativo em cada dia.

## ▶️ Como Executar Localmente (Sem Docker)
1. Criar ambiente virtual
```bash
python -m venv venv
source venv/bin/activate
```

2. Instalar dependências
```bash
pip install -r requirements.txt
```

3. Treinar o modelo
```bash
python main.py
```
## Executar a API Localmente
```bash
uvicorn api.main:app --reload
```

Acesse:
- Swagger: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

### 📬 Exemplo de Requisição /predict
```json
{
  "prices": [
    215.1, 215.8, 216.3, 216.9, 217.4,
    218.0, 218.5, 219.1, 219.6, 220.2,
    220.8, 221.3, 221.9, 222.4, 223.0,
    223.5, 224.1, 224.6, 225.2, 225.7,
    226.3, 226.8, 227.4, 227.9, 228.5,
    229.0, 229.6, 230.1, 230.7, 231.2,
    231.8, 232.3, 232.9, 233.4, 234.0,
    234.5, 235.1, 235.6, 236.2, 236.7,
    237.3, 237.8, 238.4, 238.9, 239.5,
    229.0, 229.6, 230.1, 230.7, 231.2,
    240.0, 240.6, 241.1, 241.7, 242.2,
    242.8, 243.3, 243.9, 244.4, 245.0
  ]
}
```

## 🐳 Executar com Docker
Build da imagem
```bash
docker build -t stock-prediction-api .
```

Executar container
```bash
docker run -p 8000:8000 stock-prediction-api
```

Acesse:
- http://localhost:8000/docs

## 📊 Monitoramento

- Logs salvos em logs/api.log
- Tempo de resposta monitorado via middleware
- Endpoint /health para verificação de status

## ⚠️ Escopo e Limitações

Este modelo foi desenvolvido com fins educacionais e demonstrativos,
como parte do Tech Challenge da Fase 4.

Algumas limitações importantes incluem:
- O modelo utiliza apenas o preço de fechamento (Close), não incorporando
  variáveis macroeconômicas, indicadores técnicos ou notícias.
- O modelo realiza previsão de curto prazo (próximo dia), não sendo indicado
  para previsões de longo prazo.
- O desempenho do modelo depende da estabilidade dos padrões históricos,
  podendo ser impactado por eventos inesperados de mercado.

Portanto, as previsões geradas não devem ser utilizadas como recomendação
de investimento, servindo apenas para fins acadêmicos e experimentais.

## 🔁 Generalização para Outros Ativos

A arquitetura do modelo e a API foram desenvolvidas de forma modular.
Para utilizar o sistema com outro ativo financeiro, é necessário:

1. Alterar o ticker no arquivo `main.py`
2. Realizar novo treinamento do modelo
3. Gerar novos arquivos de modelo e scaler
4. Reiniciar a API

Essa abordagem garante flexibilidade e reutilização do pipeline para
diferentes ações ou ativos financeiros.

```text
Obs.: A API foi executada localmente via Docker e FastAPI.
```
