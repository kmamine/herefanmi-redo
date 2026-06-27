# Scraper service. Build context must be the repo root (needs shared/).
FROM python:3.11-slim
WORKDIR /app

COPY shared ./shared
COPY services/scraper/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e ./shared

COPY services/scraper/hrf_scraper ./hrf_scraper

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
EXPOSE 8001
CMD ["uvicorn", "hrf_scraper.main:app", "--host", "0.0.0.0", "--port", "8001"]
