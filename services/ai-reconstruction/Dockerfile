# 1. CUDA와 PyTorch가 미리 설치된 이미지 사용 (빌드 도구 포함)
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# 🔥 [핵심] 빌드 과정에서 사용자 입력을 요구하지 않도록 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# CUDA 환경변수 설정
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

WORKDIR /app

# 1. 일반 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 2. LGM 소스코드 복사
COPY lgm_core/ ./lgm_core/

ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"

# 3. 3DGS 라이브러리 빌드 (--no-build-isolation 필수!)
RUN pip install --no-build-isolation git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git

# 나머지 소스코드 복사
COPY . .

# CMD 실행
CMD ["python3", "-u", "main.py"]