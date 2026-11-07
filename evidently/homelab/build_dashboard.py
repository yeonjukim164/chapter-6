"""책 build_evidently_dashboard.py 를 Evidently 0.7.23 API 로 옮김: income-classifier 프로젝트 + 패널
실행: kubectl -n mlops exec -i deploy/evidently-ui -- python - < evidently/homelab/build_dashboard.py"""
from evidently.sdk.models import PanelMetric
from evidently.sdk.panels import bar_plot_panel, counter_panel, line_plot_panel, text_panel
from evidently.ui.workspace import RemoteWorkspace

ws = RemoteWorkspace("http://localhost:8000")
found = ws.search_project("income-classifier")
project = found[0] if found else ws.create_project("income-classifier", description="Used to classify users into multiple income bands")
share = PanelMetric(legend="share", metric="DriftedColumnsCount", metric_labels={"value_type": "share"})
panels = [
    text_panel(title="Income Classifier Data Drift Dashboard"),
    counter_panel(title="Model Calls", values=[PanelMetric(legend="count", metric="RowCount")], size="half", aggregation="sum"),
    counter_panel(title="Share of Drifted Features", values=[share], size="half", aggregation="last"),
    line_plot_panel(title="Dataset Drift Share", values=[share]),
] + [
    bar_plot_panel(title=c, values=[PanelMetric(legend="Drift Score", metric="ValueDrift", metric_labels={"column": c})], size="half")
    for c in ["Education", "Marital-Status", "Native_country", "Occupation", "Race", "Relationship", "Sex", "Workclass"]
]
for p in panels:
    project.dashboard.add_panel(p)
print("project", project.id, "| panels", len(ws.get_dashboard(project.id).panels))
