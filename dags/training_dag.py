from airflow.models import DAG
from datetime import datetime
from airflow.operators.python import PythonOperator
from utils import (
    preprocess_data,
    get_training_features,
    train_classifier,
    get_train_test_split,
    get_evaluate
    )

with DAG(
    dag_id='ml_pipeline_training',
    schedule_interval='@daily',
    start_date=datetime(2024,8,17),
    catchup=False

) as dag:
    task_prep_data=PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    task_feature_data=PythonOperator(
        task_id='feature_data',
        python_callable=get_training_features,
    )

    task_train_test_split = PythonOperator(
        task_id='train_test_split',
        python_callable = get_train_test_split,
    )

    task_training = PythonOperator(
        task_id='train_classifier',
        python_callable = train_classifier,
    )  

    task_test_eval = PythonOperator(
        task_id='evaluate_classifier',
        python_callable = get_evaluate,
    ) 


    task_prep_data >> task_feature_data >> task_train_test_split >> task_training >> task_test_eval
