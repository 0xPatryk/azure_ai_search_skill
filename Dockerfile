FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
VOLUME ["/data/docs/in", "/data/docs/out"]
CMD ["python", "main.py"]
