# 1. Base Image: C++ 빌드 환경
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# 2. 필수 패키지 및 라이브러리 설치 (pkg-config 추가됨!)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libzmq3-dev \
    flatbuffers-compiler \
    libflatbuffers-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. 소스 코드 및 빌드 설정 복사
COPY CMakeLists.txt .
COPY src/ ./src/

# 4. CMake 빌드
#RUN mkdir build && cd build && cmake .. && make

# 5. 실행
CMD ["sh", "-c", "mkdir -p build && cd build && cmake .. && make && ./logic_core"]