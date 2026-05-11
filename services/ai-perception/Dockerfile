# services/ai-perception/Dockerfile

# 1. Base Image: 순수 Python (모든 OS, 하드웨어에서 동작)
FROM python:3.10-slim

# 2. 시스템 의존성 설치 (libgl1 추가됨!)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. 파이썬 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스코드 복사
COPY main.py .
COPY modules/ ./modules/

CMD ["python", "main.py"]