import logging
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from utils import Config, preprocess_data
from diagnostics import model_predictions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function for reporting


def score_model(test_data_path, model_file_path):
    """
    Generating confusion matrix of the model and save to disk

    Args:
        model_file_path (str): path to the saved model pickle file
        test_data_path (str): test data file path
    """
    y_pred = model_predictions(model_file_path, test_data_path)

    X, y = preprocess_data(pd.read_csv(test_data_path))
    cm = metrics.confusion_matrix(y, y_pred)
    cm_file_path = os.path.join(
        os.path.dirname(model_file_path),
        "confusionmatrix.png")
    logger.info(
        "[Reporting] Generating confusion matrix and save to {}".format(cm_file_path))
    classes = ["0", "1"]
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    cm_plot = sns.heatmap(df_cm, annot=True)
    cm_plot.figure.savefig(cm_file_path)


if __name__ == '__main__':
    cfg = Config()
    model_file_path = os.path.join(cfg.output_model_path, "trainedmodel.pkl")
    test_data_path = os.path.join(cfg.test_data_path, "testdata.csv")

    score_model(test_data_path, model_file_path)
