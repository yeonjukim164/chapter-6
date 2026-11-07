"""KServe Predictor: 4장에서 MLflow 로 저장한 sklearn 모델로 예측한다 (책의 BentoML 러너 역할).

모델 파일은 InferenceService 의 STORAGE_URI(s3://...)를 KServe storage-initializer 가
SeaweedFS 에서 /mnt/models 로 내려받아 둔다. 입력은 Transformer 가 만든 더미 컬럼 행 목록이다.
"""
import argparse

import kserve
import mlflow.sklearn
import pandas as pd


class IncomeClassifier(kserve.Model):
    def __init__(self, name: str, model_dir: str):
        super().__init__(name)
        self.model_dir = model_dir
        self.load()

    def load(self) -> bool:
        self.model = mlflow.sklearn.load_model(self.model_dir)
        self.ready = True
        return self.ready

    def predict(self, payload, headers=None, response_headers=None):
        rows = pd.DataFrame(payload["instances"])  # 행마다 학습 때와 같은 컬럼 이름을 가진 dict
        return {"predictions": self.model.predict(rows).astype(int).tolist()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
    parser.add_argument("--model_dir", default="/mnt/models")
    args, _ = parser.parse_known_args()
    kserve.ModelServer().start([IncomeClassifier(args.model_name, args.model_dir)])
