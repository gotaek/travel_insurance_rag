# --- base image ---
ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app

WORKDIR ${APP_HOME}

# 시스템 의존성 설치 (eval 폴더 최적화를 위한 추가 패키지 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ make \
    zlib1g-dev libjpeg62-turbo-dev libopenjp2-7-dev libpng-dev \
    libmagic1 ghostscript \
    git \
    wget \
    curl \
    # matplotlib/seaborn을 위한 추가 의존성
    libfreetype6-dev \
    libxft-dev \
    pkg-config \
    # pandas 최적화를 위한 의존성
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
&& rm -rf /var/lib/apt/lists/*

# 의존성 먼저 복사/설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 복사 (eval 폴더 포함)
COPY app app
COPY eval eval
COPY graph graph
COPY retriever retriever
COPY config config
COPY data data
# .env는 선택적이므로 여기서는 복사하지 않음

# 비루트 사용자 생성 및 권한 설정
RUN useradd -m appuser && \
    chown -R appuser:appuser ${APP_HOME}
USER appuser

# eval 폴더에 대한 쓰기 권한 확인
RUN mkdir -p ${APP_HOME}/eval/out

EXPOSE 8000

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]