FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py main.py ./

ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=http://ollama:11434
ENV CHROMADB_PATH=/app/data/chromadb

RUN mkdir -p /app/data/chromadb

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
