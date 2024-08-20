from airflow.models import DAG
from datetime import datetime
from airflow.operators.python import PythonOperator
from utils import (
    preprocess_inference_data,
    feature_engineer_inference,
    make_inference
)

with DAG(
    dag_id='ml_pipeline_inference',
    schedule_interval=None,  
    start_date=datetime(2024, 8, 19),
    catchup=False
) as dag:

    task_preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_inference_data,
        provide_context=True, 
    )

    task_feature_engineer = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineer_inference,
        provide_context=True,  
    )

    task_make_predictions = PythonOperator(
        task_id='make_inference',
        python_callable=make_inference,
        provide_context=True,  
    )

    task_preprocess_data >> task_feature_engineer >> task_make_predictions
