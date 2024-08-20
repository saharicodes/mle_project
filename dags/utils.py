import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np
import pickle

import mlflow
import mlflow.sklearn
# from mlflow import MlflowClient


MLFLOW_TRACKING_URI = "http://mlflow-server:5000"  
MLFLOW_EXPERIMENT_NAME = "ml_pipeline_demo"

def handle_missing_values(df: pd.DataFrame, numerical_cols: list, categorical_cols: list) -> pd.DataFrame: 
    for col in ['Saving accounts', 'Checking account']:
        df[col] = df[col].fillna('none')
    

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    numerical_cols = ['Age', 'Credit amount', 'Duration']
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    return df

def get_outliers(df, feature, iqr_threshold=1.5):
    q1 = np.percentile(df[feature], 25)
    q3 = np.percentile(df[feature], 75)
    iqr = q3 - q1
    return df.loc[(df[feature] < (q1 - iqr_threshold * iqr)) | (df[feature] > (q3 + iqr_threshold * iqr))].index

def preprocess_data():

    df = pd.read_csv("./data/german_credit_data.csv", index_col=0)
    # handle missing
    numerical_cols = ['Age', 'Credit amount', 'Duration']
    categorical_cols = ["Sex", "Job", "Housing", "Purpose", "Risk"]
    df = handle_missing_values(df, numerical_cols, categorical_cols)

    # handle outliers
    outlier_indices = set()
    for feature in numerical_cols:
        outliers = get_outliers(df, feature)
        if len(outliers) > 0:
            print("%s: %d" % (feature, len(outliers)))
            outlier_indices.update(outliers)

    df_cleaned = df.drop(index=outlier_indices)
    df_cleaned.to_csv('preproc_df.csv', index=False)

def get_training_features():

    df= pd.read_csv(f'preproc_df.csv')

    df["Age_gt_median"] = df["Age"].map(lambda x: (x >= df["Age"].median()).astype(int))
    df["Duration_gt_median"] = df["Duration"].map(lambda x: (x >= df["Duration"].median()).astype(int))
    df["Credit_amount_gt_median"] = df["Credit amount"].map(lambda x: (x >= df["Credit amount"].median()).astype(int))

    category_mapping = {
    'Sex': {'male': 1, 'female': 0},
    'Housing': {'own': 0, 'rent': 1, 'free': 2},
    'Saving accounts': {'little': 0, 'moderate': 1 , 'quite rich': 2, 'rich': 3, 'none': 4},
    'Checking account': {'little': 0, 'moderate': 1 , 'rich': 2, 'none': 3},
    'Purpose': {'radio/TV': 0, 'education': 1 , 'furniture/equipment': 2, 'car': 3, 'business': 4, 'domestic appliances': 5, 'repairs': 6, 'vacation/others': 7},
    }

    for col, mapping in category_mapping.items():
        df[col] = df[col].map(mapping)

    df['Risk'] = df['Risk'].map({'bad': 1, 'good': 0})
    df.to_csv('train_feat_df.csv', index=False)

def get_train_test_split():

    feat_df= pd.read_csv(f'train_feat_df.csv')

    X = feat_df.drop(columns=['Risk'])
    y = feat_df['Risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

def train_classifier():
    X_train = np.load('X_train.npy', allow_pickle=True)
    y_train = np.load('y_train.npy', allow_pickle=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_xgb_params = {'learning_rate': 0.3022074934689608, 
                       'max_depth': 2, 
                       'gamma': 0.284001359895123, 
                       'colsample_bytree': 0.4380467919488167, 
                       'subsample': 0.7501553939899965, 
                       'n_estimators': 2402}

    label_encode_features = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]

    with mlflow.start_run():
        classifier = XGBClassifier(**best_xgb_params, cat_features=label_encode_features)
        classifier.fit(X_train, y_train)
        mlflow.sklearn.log_model(classifier, artifact_path="model", registered_model_name="classifier_model")


def get_evaluate():
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    model_name = 'classifier_model'
    model_version_alias = 'challenger'
    model_uri = f"models:/{model_name}@{model_version_alias}"
    logistic_reg_model = mlflow.pyfunc.load_model(model_uri=model_uri)


    X_test = np.load('X_test.npy', allow_pickle=True)
    y_test = np.load('y_test.npy', allow_pickle=True)
    y_pred = logistic_reg_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('accuracy is: ', acc)


def preprocess_inference_data(**kwargs):

    input_data = kwargs['dag_run'].conf.get('input_data')
    df = pd.DataFrame(input_data)

    numerical_cols = ['Age', 'Credit amount', 'Duration']
    categorical_cols = ["Sex", "Job", "Housing", "Purpose"]

    df = handle_missing_values(df, numerical_cols, categorical_cols)

    kwargs['ti'].xcom_push(key='preprocessed_data', value=df.to_dict())

def feature_engineer_inference(**kwargs):

    preprocessed_data = kwargs['ti'].xcom_pull(key='preprocessed_data', task_ids='preprocess_data')
    df = pd.DataFrame(preprocessed_data)

    df["Age_gt_median"] = df["Age"].map(lambda x: (x >= df["Age"].median()).astype(int))
    df["Duration_gt_median"] = df["Duration"].map(lambda x: (x >= df["Duration"].median()).astype(int))
    df["Credit_amount_gt_median"] = df["Credit amount"].map(lambda x: (x >= df["Credit amount"].median()).astype(int))
    
    category_mapping = {
    'Sex': {'male': 1, 'female': 0},
    'Housing': {'own': 0, 'rent': 1, 'free': 2},
    'Saving accounts': {'little': 0, 'moderate': 1 , 'quite rich': 2, 'rich': 3, 'none': 4},
    'Checking account': {'little': 0, 'moderate': 1 , 'rich': 2, 'none': 3},
    'Purpose': {'radio/TV': 0, 'education': 1 , 'furniture/equipment': 2, 'car': 3, 'business': 4, 'domestic appliances': 5, 'repairs': 6, 'vacation/others': 7},
    }
    for col, mapping in category_mapping.items():
        df[col] = df[col].map(mapping)  

    kwargs['ti'].xcom_push(key='feature_engineered_data', value=df.to_dict())

def make_inference(**kwargs):

    feature_engineered_data = kwargs['ti'].xcom_pull(key='feature_engineered_data', task_ids='feature_engineering')
    df = pd.DataFrame(feature_engineered_data)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    model_name = 'classifier_model'
    model_version_alias = 'challenger'
    model_uri = f"models:/{model_name}@{model_version_alias}"

    classifier_model = mlflow.pyfunc.load_model(model_uri=model_uri)
    y_pred = classifier_model.predict(df)
    kwargs['ti'].xcom_push(key='predictions', value=y_pred.tolist())
