FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements-poller.txt .
RUN pip install --no-cache-dir -r requirements-poller.txt

COPY . .

EXPOSE 5002
CMD ["python", "feedPollerApp.py"]
