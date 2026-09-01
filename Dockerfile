FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/processed/soh_model.joblib data/processed/soh_model.joblib
COPY data/processed/rul_model.joblib data/processed/rul_model.joblib

# 기본 실행 대상은 모놀리식 API. docker-compose.yml은 같은 이미지를 predict/explain/
# gateway 세 컨테이너로 나눠 command만 다르게 띄운다 (MSA 변형, README 참고).
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
