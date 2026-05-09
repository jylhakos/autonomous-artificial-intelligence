FROM python:3.9-slim

RUN pip install --no-cache-dir mlflow boto3 pymysql psycopg2-binary

EXPOSE 5000

ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0"]
