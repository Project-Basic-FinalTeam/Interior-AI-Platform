FROM python:3.10-slim
WORKDIR /app
# requirements.txt가 아직 없다면 그냥 패스하도록 처리
RUN touch requirements.txt && pip install -r requirements.txt
COPY . .
CMD ["python", "-c", "print('3DGS Service Ready (Dummy)')"]