FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY pixelcnn_lfw.onnx .

RUN mkdir -p /app/outputs

EXPOSE 7860

CMD ["python", "main.py"]