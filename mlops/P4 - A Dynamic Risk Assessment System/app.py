import logging
from pathlib import Path
import subprocess
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from utils import Config
from diagnostics import model_predictions, dataframe_summary, execution_time, outdated_packages_list, missing_data_check
from scoring import score_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

# cfg = Config()
cfg = Config(config_file=str(Path(__file__).parent / "config_prod.json"))

model_file_path = os.path.join(cfg.prod_deployment_path, "trainedmodel.pkl")
prediction_model = pickle.load(open(model_file_path, "rb"))
test_data_path = os.path.join(cfg.test_data_path, "testdata.csv")
ingested_data_path = os.path.join(cfg.output_folder_path, "finaldata.csv")

# Prediction Endpoint


@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    input_file_path = request.json.get('input_file_path')
    logger.info("[API] Making predictions")
    y_pred = model_predictions(model_file_path, input_file_path)
    return str(y_pred)

# Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def score():
    logger.info("[API] Scoring")
    res = subprocess.run(["python", "scoring.py"], capture_output=True).stdout
    return res

# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    logger.info("[API] Summary statistics")
    res = dataframe_summary(ingested_data_path)
    return res

# #######################Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    logger.info("[API] Diagnostics")
    timing = execution_time()
    missing = missing_data_check(ingested_data_path)
    dependencies = outdated_packages_list()
    res = {
        'timing': timing,
        'missing_data': missing,
        'dependencies_check': dependencies.to_dict(),
    }
    return jsonify(res)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
