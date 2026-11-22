FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends postgresql-client && apt-get clean

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN useradd --create-home apguard && chown -R apguard /app
USER apguard

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
