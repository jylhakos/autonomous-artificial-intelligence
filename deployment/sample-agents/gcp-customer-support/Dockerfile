FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py main.py ./

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
