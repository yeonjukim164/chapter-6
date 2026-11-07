"""KServe Transformer: user_id 로 모델 입력을 만들고, 예측값을 레이블로 바꾼다 (책의 BentoML API 서버 역할).

요청  {"instances": [{"user_id": "..."}]}
  → Feast 온라인 스토어(Valkey)에서 피처 조회 → 학습 때와 같은 더미 컬럼으로 변환 → Predictor 호출
응답  {"predictions": [{"user_id": "...", "income_category": "<=50K"}]}

설정은 환경변수로 받는다(키 이름은 책의 production.env 와 같음). Feast 설정 준비는 책 코드(src/feature_store.py)를 그대로 쓴다.
"""
import argparse
import os
import pickle

import kserve
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from src.constants import FEATURE_LIST
from src.data_mapper import prediction_mapper
from src.feature_store import DataStore


class IncomeTransformer(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.load()

    def load(self) -> bool:
        env = os.environ
        mlflow.set_tracking_uri(env["MLFLOW_HOST"])
        versions = MlflowClient().search_model_versions(f"name='{env['MLFLOW_MODEL_NAME']}'")
        run_id = next(v.run_id for v in versions if v.current_stage == env["MLFLOW_MODEL_STAGE"])
        path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/column_list.pkl")
        with open(path, "rb") as f:
            self.col_list = pickle.load(f)
        self.store = DataStore(
            env["MINIO_HOST"],
            env["MINIO_ACCESS_KEY"],
            env["MINIO_SECRET_KEY"],
            env["FEATURE_REGISTRY_BUCKET_NAME"],
            env["FEATURE_REGSITRY_FILE_NAME"],
            env["FEAST_S3_ENDPOINT_URL"],
            env["FEAST_REDIS_HOST"],
            env["FEAST_REDIS_PASSWORD"],
        ).init_feature_store()
        self.ready = True
        return self.ready

    def preprocess(self, payload, headers=None):
        user_ids = [row["user_id"] for row in payload["instances"]]
        features = (
            self.store.get_online_features(features=FEATURE_LIST, entity_rows=[{"user_id": u} for u in user_ids])
            .to_df()
            .drop(columns=["user_id"])
        )
        # 책의 InputMapper 는 drop_first=True 라 1건 요청이면 더미가 모두 0이 됨.
        # 범주를 모두 펼친 뒤 학습 컬럼(column_list)으로 맞추면 학습 때 뺀 기준 범주 컬럼은 reindex 가 버림
        rows = pd.get_dummies(features, dtype=float).reindex(columns=self.col_list, fill_value=0)
        return {"instances": rows.to_dict(orient="records"), "user_ids": user_ids}

    async def predict(self, payload, headers=None, response_headers=None):
        user_ids = payload.pop("user_ids")  # Predictor 에는 피처만 보냄
        result = await super().predict(payload, headers, response_headers)
        return {
            "predictions": [
                {"user_id": u, "income_category": prediction_mapper[p]}
                for u, p in zip(user_ids, result["predictions"])
            ]
        }


if __name__ == "__main__":
    args, _ = kserve.model_server.parser.parse_known_args()
    kserve.ModelServer().start([IncomeTransformer(args.model_name)])
