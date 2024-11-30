import logging
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from utils import Config, preprocess_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function for model scoring


def score_model(
        model_file_path,
        test_data_path,
        store=True
):
    """
    Score the trained model with test data and save the score in the output folder

    Args:
        model_file_path (str): path to the saved model pickle file
        test_data_path (str): test data file path

    Returns:
        f1_score (float): f1 score of the model
    """
    lr_model = pickle.load(open(model_file_path, "rb"))
    data = pd.read_csv(test_data_path)
    X, y = preprocess_data(data)
    y_pred = lr_model.predict(X)
    f1_score = metrics.f1_score(y, y_pred)

    if store:
        score_file = os.path.join(
            os.path.dirname(model_file_path),
            "latestscore.txt")
        with open(score_file, "w") as f:
            f.write(str(f1_score))
        logger.info(
            f"[Scoring] Model f1_score ({f1_score}) on test data saved to {score_file}")
    else:
        logger.info(f"[Scoring] Model f1_score ({f1_score}) on test data")

    return f1_score


if __name__ == "__main__":
    cfg = Config()
    model_file_path = os.path.join(cfg.output_model_path, "trainedmodel.pkl")
    test_data_path = os.path.join(cfg.test_data_path, "testdata.csv")
    print(score_model(model_file_path, test_data_path))
