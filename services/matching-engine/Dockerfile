FROM python:3.10-slim
WORKDIR /app
RUN touch requirements.txt && pip install -r requirements.txt
COPY . .
CMD ["python", "-c", "print('Matching Engine Ready (Dummy)')"]