import os
import logging
import time
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime

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

def get_xcom_value(dag_id, dag_run_id, task_id, key='predictions'):
    """Fetch the XCom value for a given DAG run and task."""
    xcom_url = f"{AIRFLOW_ENDPOINT_URL}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/xcomEntries/{key}"
    response = requests.get(xcom_url, **airflow_api_params)
    if response.status_code == 200:
        return response.json().get('value')
    else:
        return None

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


    @app.route('/trigger_inference/<dag_id>', methods=['POST'])
    def trigger_inference_dag(dag_id):
        input_data = request.json
        payload = {
            "conf": {
                "input_data": input_data
            }
        }

        try:
            response = requests.post(
                f"{AIRFLOW_ENDPOINT_URL}/dags/{dag_id}/dagRuns",
                json=payload,
                **airflow_api_params
            )

            if response.status_code == 200:
                dag_run_id = response.json().get('dag_run_id')

                for _ in range(10):  
                    dag_status_url = f"{AIRFLOW_ENDPOINT_URL}/dags/{dag_id}/dagRuns/{dag_run_id}"
                    status_response = requests.get(dag_status_url, **airflow_api_params)
                    if status_response.status_code == 200:
                        dag_run = status_response.json()
                        if dag_run['state'] == 'success':
                            prediction = get_xcom_value(dag_id, dag_run_id, task_id='make_inference')
                            if prediction:
                                return jsonify({"status": "success", "prediction": prediction})
                            else:
                                return jsonify({"status": "failed", "message": "Prediction XCom value not found"}), 500
                        elif dag_run['state'] in ['failed', 'up_for_retry', 'up_for_reschedule']:
                            return jsonify({"status": "failed", "message": f"DAG run {dag_run['state']}"}), 500

                    time.sleep(5)

                return jsonify({"status": "failed", "message": "Prediction not ready"}), 202 
            else:
                return response_fail("Failed to trigger Inference DAG", code=response.status_code)

        except Exception as e:
            return fail_from_error('Unable to trigger Inference DAG!')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5001)
