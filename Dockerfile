FROM python:3.10-slim-bookworm

WORKDIR /app

COPY requirements.txt .
COPY setup.py .
COPY src ./src

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "app.py"]