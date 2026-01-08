# ==============================
# IMAGEM BASE
# ==============================
FROM python:3.12-slim

# ==============================
# VARIÁVEIS DE AMBIENTE
# ==============================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==============================
# DIRETÓRIO DE TRABALHO
# ==============================
WORKDIR /app

# ==============================
# DEPENDÊNCIAS
# ==============================
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ==============================
# COPIAR O PROJETO
# ==============================
COPY . .

# ==============================
# EXPOR PORTA DA API
# ==============================
EXPOSE 8000

# ==============================
# COMANDO DE INICIALIZAÇÃO
# ==============================
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
