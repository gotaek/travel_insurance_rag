# --- base image ---
    ARG PYTHON_VERSION=3.11-slim
    FROM python:${PYTHON_VERSION} AS runtime
    
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PIP_DISABLE_PIP_VERSION_CHECK=on \
        PIP_NO_CACHE_DIR=1 \
        APP_HOME=/app
    
    WORKDIR ${APP_HOME}
    
    # healthcheck용 curl 설치
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ make \
        zlib1g-dev libjpeg62-turbo-dev libopenjp2-7-dev libpng-dev \
        libmagic1 ghostscript \
    && rm -rf /var/lib/apt/lists/*
    
    # 의존성 먼저 복사/설치
    COPY requirements.txt .
    RUN pip install -r requirements.txt
    
    # 애플리케이션 복사
    COPY app app
    # config와 .env는 선택적이므로 여기서는 복사하지 않음
    
    # 비루트 사용자
    RUN useradd -m appuser
    USER appuser
    
    EXPOSE 8000
    
    CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]