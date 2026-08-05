# Inference image: Flask, onnxruntime and the exported ONNX model.
#
# Training lives in training/ and is never installed here. Every dependency
# below ships a prebuilt wheel, so no compiler is needed and a single stage is
# enough.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Gunicorn must not run as root.
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

CMD ["gunicorn", "--chdir", "app", "-w", "1", "-b", "0.0.0.0:5000", "main:app"]
