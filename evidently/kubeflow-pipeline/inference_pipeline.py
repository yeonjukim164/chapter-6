import kfp
from kfp import dsl, compiler, kubernetes

# Load KFP v2-compatible components
fetch_data_op = kfp.components.load_component_from_file(
    "components/read_data/component.yaml"
)
retrieve_features_op = kfp.components.load_component_from_file(
    "components/retrieve_features/component.yaml"
)
detect_drift_op = kfp.components.load_component_from_file(
    "components/detect_drift/component.yaml"
)
run_inference_op = kfp.components.load_component_from_file(
    "components/run_inference/component.yaml"
)

write_data_op = kfp.components.load_component_from_file(
    "components/write_data/component.yaml"
)


@dsl.pipeline(
    name="income-classifier-pipeline",
    description="A Kubeflow pipeline to classify income categories using KFP v2",
)
def income_classifier_pipeline(
    minio_host: str,
    access_key: str,
    secret_key: str,
    entity_df_bucket: str,
    entity_df_filename: str,
    feature_store_bucket_name: str,
    feature_store_config_file_name: str,
    reference_dataset_name: str,
    evidently_workspace_url: str,
    evidently_ui_project_name: str,
    feature_list: str,
    model_name: str,
    model_type: str,
    model_stage: str,
    mlflow_host: str,
    output_bucket: str,
    output_file_name: str,
):
    # Fetch data task
    fetch_data_task = fetch_data_op(
        minio_host=minio_host,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=entity_df_bucket,
        file_name=entity_df_filename,
    )
    fetch_data_task.set_caching_options(enable_caching=False)

    # Retrieve features task
    retrieve_features_task = retrieve_features_op(
        minio_host=minio_host,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=feature_store_bucket_name,
        file_name=feature_store_config_file_name,
        entity_df=fetch_data_task.outputs["data_output"],  # Reference to task output
        feature_list=feature_list,
    )
    detect_drift_task = detect_drift_op(
        model_name=model_name,
        model_stage=model_stage,
        mlflow_host=mlflow_host,
        minio_host=minio_host,
        access_key=access_key,
        secret_key=secret_key,
        reference_dataset_name=reference_dataset_name,
        feature_dataset_path=retrieve_features_task.outputs["data_output"],
        evidently_workspace_url=evidently_workspace_url,
        evidently_ui_project_name=evidently_ui_project_name,
    )

    run_inference_task = run_inference_op(
        minio_host=minio_host,
        access_key=access_key,
        secret_key=secret_key,
        model_name=model_name,
        model_type=model_type,
        model_stage=model_stage,
        mlflow_host=mlflow_host,
        input_data=retrieve_features_task.outputs["data_output"],
    ).after(detect_drift_task)

    # Write data task
    write_data_task = write_data_op(
        minio_host=minio_host,
        access_key=access_key,
        secret_key=secret_key,
        bucket_name=output_bucket,
        file_name=output_file_name,
        input_data=run_inference_task.outputs["data_output"],
    )
    # 컴포넌트 이미지는 GHCR 비공개 패키지 → 사용자 네임스페이스의 ghcr-pull Secret 으로 받음
    for task in (fetch_data_task, retrieve_features_task, detect_drift_task, run_inference_task, write_data_task):
        kubernetes.set_image_pull_secrets(task, ["ghcr-pull"])


# Compile pipeline to YAML
compiler.Compiler().compile(
    pipeline_func=income_classifier_pipeline,
    package_path="income_classifier_pipeline_with_drift.yaml",
)
