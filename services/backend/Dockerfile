# Backend orchestrator. Build context must be the repo root (needs shared/).
FROM python:3.11-slim
WORKDIR /app

COPY shared ./shared
COPY services/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e ./shared

COPY services/backend/app ./app

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
EXPOSE 10000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
