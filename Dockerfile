# Stage 1: Build React frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
ENV CI=false
RUN npm run build

# Stage 2: Python backend
FROM python:3.10-slim AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY backend/ ./backend/
COPY --from=frontend-builder /app/frontend/build ./backend/static/
RUN useradd -m appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]