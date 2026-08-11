FROM python:3.10-slim

WORKDIR /app

COPY requirements-prod.txt .

RUN pip install --no-cache-dir -r requirements-prod.txt

COPY . .

ENV PYTHONPATH=/app/src

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]