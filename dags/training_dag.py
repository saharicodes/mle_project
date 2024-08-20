from airflow.models import DAG
from datetime import datetime
from airflow.operators.python import PythonOperator
import pandas as pd 
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np
import pickle

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from utils import (
    preprocess_data,
    get_training_features,
    train_classifier,
    get_train_test_split,
    get_evaluate
    )

# MLFLOW_TRACKING_URI = "http://mlflow-server:5000"  
# MLFLOW_EXPERIMENT_NAME = "ml_pipeline_demo"

# def handle_missing_values(df: pd.DataFrame, numerical_cols: list, categorical_cols: list) -> pd.DataFrame: 
#     for col in ['Saving accounts', 'Checking account']:
#         df[col] = df[col].fillna('none')
    

#     for col in categorical_cols:
#         df[col] = df[col].fillna(df[col].mode()[0])
    
#     numerical_cols = ['Age', 'Credit amount', 'Duration']
#     for col in numerical_cols:
#         df[col] = df[col].fillna(df[col].median())
    
#     return df

# def get_outliers(df, feature, iqr_threshold=1.5):
#     q1 = np.percentile(df[feature], 25)
#     q3 = np.percentile(df[feature], 75)
#     iqr = q3 - q1
#     return df.loc[(df[feature] < (q1 - iqr_threshold * iqr)) | (df[feature] > (q3 + iqr_threshold * iqr))].index

# def preprocess_data():

#     df = pd.read_csv("../data/german_credit_data.csv")
#     # handle missing
#     numerical_cols = ['Age', 'Credit amount', 'Duration']
#     categorical_cols = ["Sex", "Job", "Housing", "Purpose", "Risk"]
#     df = handle_missing_values(df, numerical_cols, categorical_cols)

#     # handle outliers
#     outlier_indices = set()
#     for feature in numerical_cols:
#         outliers = get_outliers(df, feature)
#         if len(outliers) > 0:
#             print("%s: %d" % (feature, len(outliers)))
#             outlier_indices.update(outliers)

#     df_cleaned = df.drop(index=outlier_indices)
#     df_cleaned.to_csv('preproc_df.csv', index=False)

# def get_training_features():

#     df= pd.read_csv(f'preproc_df.csv')

#     df["Age_gt_median"] = df["Age"].map(lambda x: (x >= df["Age"].median()).astype(int))
#     df["Duration_gt_median"] = df["Duration"].map(lambda x: (x >= df["Duration"].median()).astype(int))
#     df["Credit_amount_gt_median"] = df["Credit amount"].map(lambda x: (x >= df["Credit amount"].median()).astype(int))

#     label_encoder = LabelEncoder()
#     label_encode_features = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]

#     for feature in label_encode_features:
#         df[feature] = label_encoder.fit_transform(df[feature])

#     df['Risk'] = df['Risk'].map({'bad': 1, 'good': 0})
#     df.to_csv('train_feat_df.csv', index=False)


# def get_train_test_split():

#     feat_df= pd.read_csv(f'train_feat_df.csv')

#     X = feat_df.drop(columns=['Risk'])
#     y = feat_df['Risk']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#     np.save('X_train.npy', X_train)
#     np.save('X_test.npy', X_test)
#     np.save('y_train.npy', y_train)
#     np.save('y_test.npy', y_test)

# def train_classifier():
#     X_train = np.load('X_train.npy', allow_pickle=True)
#     y_train = np.load('y_train.npy', allow_pickle=True)

#     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#     mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

#     best_xgb_params = {'learning_rate': 0.3022074934689608, 
#                        'max_depth': 2, 
#                        'gamma': 0.284001359895123, 
#                        'colsample_bytree': 0.4380467919488167, 
#                        'subsample': 0.7501553939899965, 
#                        'n_estimators': 2402}

#     label_encode_features = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]

#     with mlflow.start_run():
#         classifier = XGBClassifier(**best_xgb_params, cat_features=label_encode_features)
#         classifier.fit(X_train, y_train)
#         mlflow.sklearn.log_model(classifier, artifact_path="model", registered_model_name="classifier_model")

#     with open('model.pkl', 'wb') as f:
#         pickle.dump(classifier, f)


# def get_evaluate():
#     with open('model.pkl', 'rb') as f:
#         logistic_reg_model = pickle.load(f)
    
#     # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
#     # mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
#     # # model_uri = "runs:/9a81b2b206864265bf8650689eaa35f3/model"
#     # model_name = 'classifier_model'
#     # model_version_alias = 'challenger'
#     # model_uri = f"models:/{model_name}@{model_version_alias}"
#     # print('model uri', model_uri)
#     # # logistic_reg_model = mlflow.sklearn.load_model(model_uri=model_uri)
#     # logistic_reg_model = mlflow.pyfunc.load_model(model_uri=model_uri)


#     X_test = np.load('X_test.npy', allow_pickle=True)
#     y_test = np.load('y_test.npy', allow_pickle=True)
#     y_pred = logistic_reg_model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print('accuracy is: ', acc)

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
