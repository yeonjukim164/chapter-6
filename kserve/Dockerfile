# KServe Predictor / Transformer 공용 이미지 (빌드 컨텍스트: 저장소 루트)
#   Predictor:   python predictor.py --model_name <InferenceService 이름>
#   Transformer: python transformer.py  (KServe 가 --model_name, --predictor_host 를 붙여 줌)
FROM python:3.12-slim
LABEL org.opencontainers.image.source=https://github.com/yeonjukim164/chapter-6
WORKDIR /app
COPY kserve/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY bentoml/src /app/src
COPY kserve/predictor.py kserve/transformer.py /app/
ENV PYTHONUNBUFFERED=1
