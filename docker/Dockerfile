FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

# retry up to three times with a short sleep
RUN set -eux; \
    for i in 1 2 3; do \
        pip install --no-cache-dir -r requirements.txt && break || \
        (sleep 5 && echo "retrying pip…"); \
    done

COPY app/ app/
COPY data/ data/
COPY .env .env

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]