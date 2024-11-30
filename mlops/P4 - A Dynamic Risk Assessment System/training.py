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

# Function for training the model


def train_model(
    dataset_csv_path,
    model_folder_path
):
    """
    Train the model and save to the output folder

    Args:
        dataset_csv_path (str): path to the dataset csv file
        model_folder_path (str): path to the output model folder
    """

    lr_model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    # fit the logistic regression to your data

    data = pd.read_csv(dataset_csv_path)
    X, y = preprocess_data(data)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=37)
    # X_train, y_train = X, y
    lr_model.fit(X_train, y_train)
    # write the trained model to your workspace in a file called
    # trainedmodel.pkl
    os.makedirs(model_folder_path, exist_ok=True)
    model_file_path = os.path.join(model_folder_path, "trainedmodel.pkl")
    with open(model_file_path, "wb") as f:
        pickle.dump(lr_model, f)
    logger.info("[Training] Trained model saved to {}".format(model_file_path))


if __name__ == "__main__":
    config = Config()
    dataset_csv_path = os.path.join(config.output_folder_path, "finaldata.csv")
    model_folder_path = config.output_model_path
    train_model(dataset_csv_path, model_folder_path)
