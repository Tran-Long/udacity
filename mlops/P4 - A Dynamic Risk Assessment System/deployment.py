import logging
import shutil
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from utils import Config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# function for deployment


def store_model_into_pickle(
        file_paths,
        prod_deployment_path,
):
    """
    Transfer artifacts to deployment environment

    Args:
        file_paths (list): list file paths to be copied to deployment
        prod_deployment_path (str): production deployment path

    """
    logger.info(
        "[Deployment] Transferring artifacts to deployment environment")
    os.makedirs(prod_deployment_path, exist_ok=True)
    for file in file_paths:
        shutil.copy(file, prod_deployment_path)


if __name__ == "__main__":
    cfg = Config()
    model_file_path = os.path.join(cfg.output_model_path, "trainedmodel.pkl")
    score_file_path = os.path.join(cfg.output_model_path, "latestscore.txt")
    ingest_file_path = os.path.join(
        cfg.output_folder_path,
        "ingestedfiles.txt")
    prod_deployment_path = cfg.prod_deployment_path
    store_model_into_pickle(
        [model_file_path, score_file_path, ingest_file_path],
        prod_deployment_path
    )
