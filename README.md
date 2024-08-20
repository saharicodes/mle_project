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

1. Clone the Repository:

   ```bash
   git clone https://github.com/saharicodes/mle_project.git
   cd mle_project
   ```
2. Create `.env` file and run 
   ```bash
   echo -e "AIRFLOW_UID=$(id -u)" > .env
   ```
2. Run the following command to initialize the Airflow metadata database:

   ```bash
    docker-compose up airflow-init
   ```

3. Build the Docker images and start the containers
   ```bash
    docker-compose up --build
   ```

This command will start the following services:

* Airflow Webserver: Accessible at http://localhost:8080
* Airflow Scheduler
* Airflow Worker
* Airflow Triggerer
* Airflow redis #cache
* Airflow postgres #backend store
* MLflow Server: Accessible at http://localhost:5000
* Flask API: Accessible at http://localhost:5001

## Usage

### Triggering DAGs

#### Trigger Inference DAG via Flask API

Use a tool like Postman, curl, or any HTTP client to send a POST request with the input data to trigger the inference DAG.

```bash
curl -X POST http://localhost:5001/trigger_inference/ml_pipeline_inference \
-H "Content-Type: application/json" \
-d '{"Age": [67],
 "Sex": ["male"],
 "Job": [2],
 "Housing": ["own"],
 "Saving accounts": [null],
 "Checking account": ["little"],
 "Credit amount": [1169],
 "Duration": [6],
 "Purpose": ["radio/TV"]}'
```
Sample output format 
```bash
{
    "prediction": "[0]",
    "status": "success"
}
```
You can also use Postman client running on a host machine to trigger API endpoints

#### Trigger Training DAG via Flask API
The training DAG can be triggered similarly:

```bash
curl -X POST http://localhost:5001/trigger/ml_pipeline_training \
-H "Content-Type: application/json"
```

**Note**: You can also use Postman client running on a host machine to trigger API endpoints. 

### Access Airflow and Mlflow Web UI
Access the Airflow web interface at http://localhost:8080 to manually trigger DAGs or monitor their status.

You can also access the Mlflow web interface at http://localhost:5000 to view experiments and registered models and add model ailias for the well performing model to be used for inference.

### Stopping the Services
To stop all running services, use:

``` bash
docker-compose down
```