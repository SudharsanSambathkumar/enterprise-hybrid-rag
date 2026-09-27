FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "hybrid_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
