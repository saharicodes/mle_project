# Airflow Training and Inference Pipeline with MLflow and Flask API

This project integrates Apache Airflow with a Flask API to manage both the training and inference of a machine learning model used for predicting credit risk. The model development process, including exploratory data analysis (EDA), is documented in a Jupyter notebook. Model artifacts are registered in MLflow, enabling version control and seamless retrieval during inference.

## Project Structure
```
├── dags    # Airflow DAGs and utilities
│ ├── inference_dag.py      # Inference DAG definition
│ ├── training_dag.py       # Training DAG definition
│ └── utils.py      # Utility functions for data processing 
├── flask_app 
│ ├── app.py 
│ └── requirements.txt      # Flask dependencies
├── data # Data directory
│ └── german_credit_data.csv    # Sample dataset
├── notebooks   
│ └── credit_risk_eda.ipynb     # EDA and model development 
├── docker-compose.yaml
├── Dockerfile.airflow
├── Dockerfile.mlflow
├── Dockerfile.flask
└── README.md 
```

## Features

- **Training and Inference Pipelines**: The project includes both training and inference DAGs, allowing for comprehensive model lifecycle management.
- **ML Model for Credit Risk Prediction**: The machine learning model is a classifier used to predict credit risk.
- **Exploratory Data Analysis (EDA)**: EDA and the model development process are documented in a Jupyter notebook (`credit_risk_eda.ipynb`).
- **MLflow Integration**: Model artifacts are registered in MLflow, enabling version control and easy retrieval during inference with the appropriate model alias.
- **Flexible Training Triggers**: The training DAG can be triggered manually via the Airflow web UI, scheduled to run at specific intervals, or triggered using a Flask endpoint.
- **Prediction Retrieval**: Predictions are stored in Airflow's XCom and can be retrieved via the inference Flask API endpoint.

## Prerequisites

- Python 3.11+
- Apache Airflow 2.x
- Docker (for containerization)
- Redis (for task queue management)
- MLflow (for model management)
- Flask API (inference and training endpoint)

## Installation and Setup

1. **Clone the Repository**:

   ```bash
