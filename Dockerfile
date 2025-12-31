FROM python:3.12-slim

WORKDIR /app

# deps
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# code
COPY app /app/app

EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
