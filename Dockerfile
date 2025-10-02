# HAFİF, SAĞLAM: Python 3.11 + FAISS için libgomp
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# FAISS bağımlılığı
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# Önce CPU sürümü Torch’u kur → CUDA paketleri gelmez, RAM taşmaz
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install --no-cache-dir -r requirements.txt


# Uygulama dosyaları
COPY . .

# Varsayılan ortam değişkenleri (deploy'da override edebilirsin)
ENV DOCS_DIR=/app/docs \
    INDEX_PATH=/app/faiss.index \
    META_PATH=/app/meta.pkl \
    BOT_TITLE="Fırat Üniversitesi Chatbot"

EXPOSE 8000
CMD ["uvicorn","proje:app","--host","0.0.0.0","--port","8000"]
