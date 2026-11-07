"""KServe Transformer: user_id 로 모델 입력을 만들고, 예측값을 레이블로 바꾼다 (책의 BentoML API 서버 역할).

요청  {"instances": [{"user_id": "..."}]}
  → Feast 온라인 스토어(Valkey)에서 피처 조회 → 학습 때와 같은 더미 컬럼으로 변환 → Predictor 호출
응답  {"predictions": [{"user_id": "...", "income_category": "<=50K"}]}

설정은 환경변수로 받는다(키 이름은 책의 production.env 와 같음). Feast 설정 준비는 책 코드(src/feature_store.py)를 그대로 쓴다.
"""
import argparse
import json
import logging
import os
import pickle

import kserve
import mlflow
import pandas as pd
from evidently import Report
from evidently.metrics import DriftedColumnsCount, RowCount, ValueDrift
from evidently.ui.workspace import RemoteWorkspace
from minio import Minio
from mlflow.tracking import MlflowClient

from src.constants import FEATURE_LIST
from src.data_mapper import prediction_mapper
from src.feature_store import DataStore

logger = logging.getLogger(__name__)
DRIFT_COLUMNS = ["Workclass", "Education", "Marital-Status", "Occupation", "Relationship", "Race", "Sex", "Native_country"]


class MonitoringService:
    """책 리스트 6.10: 최근 요청 window_size 건(마이크로배치)을 참조 데이터와 비교해 Evidently 프로젝트에 기록 (Evidently 0.7 API)"""

    def __init__(self, reference: pd.DataFrame, workspace: RemoteWorkspace, project_id, window_size: int):
        self.reference, self.workspace, self.project_id = reference, workspace, project_id
        self.window_size = int(window_size)
        self.report = Report([RowCount(), DriftedColumnsCount()] + [ValueDrift(column=c) for c in DRIFT_COLUMNS])
        self.current = pd.DataFrame()

    def iterate(self, new_rows: pd.DataFrame):
        self.current = pd.concat([self.current, new_rows], ignore_index=True).iloc[-self.window_size:]
        if len(self.current) < self.window_size:
            logger.info(f"Not enough data for measurement: {len(self.current)} of {self.window_size}. Waiting more data")
            return
        logger.info("Running report")
        snapshot = self.report.run(reference_data=self.reference, current_data=self.current.reset_index(drop=True))
        self.workspace.add_run(self.project_id, snapshot)


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
        self.monitor = self._init_monitoring(run_id) if env.get("EVIDENTLY_WORKSPACE_URL") else None
        self.ready = True
        return self.ready

    def _init_monitoring(self, run_id: str) -> MonitoringService:
        """책 리스트 6.11: MLflow run 에 기록된 참조 데이터셋을 받아 MonitoringService 를 만든다"""
        env = os.environ
        run = mlflow.get_run(run_id)
        uri = next(
            json.loads(d.dataset.source)["uri"]
            for d in run.inputs.dataset_inputs
            if any(t.value == env["REFERENCE_DATASET_NAME"] for t in d.tags)
        )
        bucket, key = uri.split("/", 1)
        Minio(env["MINIO_HOST"], env["MINIO_ACCESS_KEY"], env["MINIO_SECRET_KEY"], secure=False).fget_object(bucket, key, "/tmp/reference.csv")
        workspace = RemoteWorkspace(env["EVIDENTLY_WORKSPACE_URL"])
        found = workspace.search_project(env["EVIDENTLY_PROJECT_NAME"])
        project = found[0] if found else workspace.create_project(env["EVIDENTLY_PROJECT_NAME"])
        return MonitoringService(pd.read_csv("/tmp/reference.csv")[DRIFT_COLUMNS], workspace, project.id,
                                 env.get("EVIDENTLY_REPORT_WINDOW_SIZE", "100"))

    def preprocess(self, payload, headers=None):
        user_ids = [row["user_id"] for row in payload["instances"]]
        features = (
            self.store.get_online_features(features=FEATURE_LIST, entity_rows=[{"user_id": u} for u in user_ids])
            .to_df()
            .drop(columns=["user_id"])
        )
        if self.monitor:  # 6.2.3 실시간 드리프트: 조회한 피처를 윈도에 쌓고, 다 차면 보고서 실행
            self.monitor.iterate(features[DRIFT_COLUMNS])
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
