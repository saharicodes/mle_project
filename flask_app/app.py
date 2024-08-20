import os
import logging
import requests
from flask import Flask, jsonify
from flask_cors import CORS

AIRFLOW_ENDPOINT_URL = "http://airflow-webserver:8080/api/v1"

airflow_api_params = {
    'headers': {
        'Content-type': 'application/json',
        'Accept': 'application/json'
    },
    'auth': requests.auth.HTTPBasicAuth(
        os.getenv('_AIRFLOW_WWW_USER_USERNAME', 'airflow'),
        os.getenv('_AIRFLOW_WWW_USER_PASSWORD', 'airflow')
    )
}

def response_fail(msg, code=500):
    """
    Uniform way to return a "failed" response as JSON
    """
    return jsonify({'status': 'failed', 'message': msg}), code

def fail_from_error(msg):
    """
    Streamline code for returning a failed response from within 
    an `except` statement
    """
    logging.exception(msg)
    return response_fail(msg)

def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def home():
        return "Welcome to the Airflow DAG Trigger API!"
    
    @app.route("/trigger/<dag_id>", methods=['POST'])
    def trigger_training_dag(dag_id):
        """
        Trigger a DAG run. Un-pause the DAG first in case it is 
        paused.
        """
        # dag_id = "ml_pipeline_training"
        try:
            resp = requests.patch(
                f"{AIRFLOW_ENDPOINT_URL}/dags/{dag_id}",
                json={'is_paused': False},
                **airflow_api_params
            )
            if resp.status_code != 200:
                return response_fail('Unable to un-pause DAG!', code=resp.status_code)
            print(repr(resp.json()))
        except Exception as e:
            return fail_from_error('Unable to un-pause DAG!')
        
        try:
            resp = requests.post(
                f"{AIRFLOW_ENDPOINT_URL}/dags/{dag_id}/dagRuns",
                json={'conf': {}},  # parameter
                **airflow_api_params
            )
            if resp.status_code != 200:
                return response_fail('Unable to trigger DAG!', code=resp.status_code)
        except Exception as e:
            return fail_from_error('Unable to trigger DAG!')
        
        return jsonify({"status": "success", "dag_run_id": resp.json().get('dag_run_id')})

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5001)
