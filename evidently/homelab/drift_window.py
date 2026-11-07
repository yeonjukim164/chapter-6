"""같은 데이터(4장 참조 데이터셋)에서 뽑은 표본도 윈도가 작으면 드리프트로 판정되는지 확인 (Evidently 최신 API)"""
import os, re, warnings
warnings.filterwarnings("ignore")
import pandas as pd
from minio import Minio
import evidently
from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift

Minio("seaweedfs.kubeflow.svc.cluster.local:9000", os.environ["AWS_ACCESS_KEY_ID"], os.environ["AWS_SECRET_ACCESS_KEY"], secure=False) \
    .fget_object("mlflow-datasets", "income-classifier-datasets/feature_df-213b52da659c44d497996eeba8be1503.csv", "/tmp/reference.csv")
cols = ["Workclass", "Education", "Marital-Status", "Occupation", "Relationship", "Race", "Sex", "Native_country"]
ref = pd.read_csv("/tmp/reference.csv")[cols]
print("evidently", evidently.__version__, "| pandas", pd.__version__, "| reference:", ref.shape)
report = Report([DriftedColumnsCount()] + [ValueDrift(column=c) for c in cols])
first = True
for n in (50, 100, 500, 1000, 5000):
    shares, detected, methods = [], 0, set()
    for seed in range(5):
        metrics = report.run(current_data=ref.sample(n=n, random_state=seed), reference_data=ref).dict()["metrics"]
        if first:
            print("metric keys:", list(metrics[0]), "| names:", [m.get("metric_name") or m.get("metric_id") for m in metrics][:2]); first = False
        dc = next(m for m in metrics if str(m.get("metric_name") or m.get("metric_id")).startswith("DriftedColumnsCount"))
        shares.append(dc["value"]["share"]); detected += dc["value"]["share"] >= 0.5
        methods |= {re.search(r"method=([^,)]+)", str(m.get("metric_name") or m.get("metric_id"))).group(1) for m in metrics if "method=" in str(m.get("metric_name") or m.get("metric_id"))}
    print(f"window {n:5d}: share_of_drifted_columns mean {sum(shares)/5:.3f} ({', '.join(f'{s:.3f}' for s in shares)}) | dataset_drift {detected}/5 | stattest {sorted(methods)}")
