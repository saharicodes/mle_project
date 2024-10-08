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

## Further Discussion

### CI/CD Pipeline Proposal
This proposal outlines the steps and tools required to implement a CI/CD pipeline for automating the deployment of the airflow ML pipline into production. The pipeline will handle code integration, testing, Docker image building, pushing to Azure ACR, and deployment to either a VM or a Kubernetes cluster.

Tools and Technologies
Version Control System: Git (GitHub, GitLab, or Bitbucket)
CI/CD Pipeline: Jenkins
Containerization: Docker
Container Registry: Azure Container Registry (ACR)
Deployment: Docker Compose (for VM) 
Testing and Linting: PyTest and Flake8

#### CI/CD Pipeline Stages
1. Continuous Integration (CI)
- The CI pipeline begins by checking out the latest code from the repository when changes are pushed or a pull request is created.
- Static Code Analysis and Linting using Flake8 to enforce coding standards.
- Run unit tests using PyTest to ensure code is not broken.
2. Building and Pushing Docker Image
- Conteinerize application using Docker
- After building, tag the Docker image and push it to ACR or any other registery for storage and later use in deployment.
3. Continuous Deployment (CD)
- Deploy the image to a `development` environment for further monitoring.
- Manually triger production deploy job to deploy the application to `production` environment, eg. on a VM using Docker Compose or on a Kubernetes cluster using Helm charts.

#### Live Performance Monitoring:

##### Metrics:

1. Prediction Accuracy: Compare model predictions against the ground truth to measure accuracy.
2. Latency: Track the time it takes for the model to generate predictions.
3. Observe changes in statistical properties of incoming data compared to training data. (Data Drift)
4. Monitor model's performance over time to detect any gradual degradation. (Model Drift)

##### Monitoring strategy:

- Store live predictions in a database (e.g., PostgreSQL, MongoDB) for performance monitoring against ground truth.
- Use Grafana to pull prediction data from the database and visualize performance metrics.
- Set up alert rules in Grafana based on business requirements (e.g., accuracy drops, latency increases).
- Scheduled or automated regular model retraining on new data to maintain accuracy.
- Use Canary deploymentstrategy to gradually roll out the new model version to a small subset of users, monitor its performance, and if no issues are detected, progressively increase the deployment to the entire user base.

## Next Steps

### 1. Unit Testing: 
- Implementing unit tests to ensure the robustness and reliability of the application.

### 2. Implement Logging

-  Logging to ensure that both your Airflow tasks and Flask API include detailed logging to capture essential information for debugging and performance monitoring.
- Consider using centralized logging solutions like graylogs to collect, analyze, and visualize your logs.

### 3. Implement CI/CD Pipeline

- Integrating a CI/CD pipeline to automate the testing, deployment, and monitoring of the Airflow DAGs and Flask API using tools like GitHub Actions, Jenkins, or GitLab CI.

### 6. Secure API

Implement authentication and authorization mechanisms to secure the API endpoints from unauthorized access.

### 7. Deploy to Production

- Deploying the application to a production environment using container orchestration tools like Kubernetes or Docker Swarm. and ensure that the services are load-balanced and highly available to handle production-level traffic.


