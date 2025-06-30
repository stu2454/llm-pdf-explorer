# ---- base stage ----------------------------------------------------------
FROM python:3.12-slim as base
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Streamlit needs this or UTF-8 breaks in Docker logs
    LC_ALL=C.UTF-8

WORKDIR /app

# ---- deps stage ----------------------------------------------------------
FROM base as deps
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- final stage ---------------------------------------------------------
FROM deps as final
# Create an unprivileged user (nobody:nogroup already exists)
USER nobody
COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0"]

