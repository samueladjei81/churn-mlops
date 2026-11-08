# Packages the API with the current champion model.
# Run `make simulate` (or `make train`) first so models/ contains a champion.
FROM python:3.12-slim

WORKDIR /app
ENV CHURN_HOME=/app PYTHONPATH=/app/src PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"

COPY config.yaml ./
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "churn.api:app", "--host", "0.0.0.0", "--port", "8000"]
