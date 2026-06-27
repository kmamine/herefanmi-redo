# LLM service — calls an OpenAI-compatible provider (vLLM Gemma by default).
# Build context must be the repo root (needs shared/).
FROM python:3.11-slim
WORKDIR /app

COPY shared ./shared
COPY services/llm/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e ./shared

COPY services/llm/hrf_llm ./hrf_llm

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
EXPOSE 5002
CMD ["uvicorn", "hrf_llm.main:app", "--host", "0.0.0.0", "--port", "5002"]
