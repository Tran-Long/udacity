
import logging
import pickle
import subprocess
import pandas as pd
import numpy as np
import timeit
import os
import json
from utils import Config, preprocess_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to get model predictions


def model_predictions(
        model_file_path,
        test_data_path
):
    """
    Make prediction with given model and data

    Args:
        model_file_path (str): path to the saved model pickle file
        test_data_path (str): test data file path

    Returns:
        y_pred (np.darray): prediction of the model
    """
    logger.info("[Prediction] Making predictions with trained model")
    lr_model = pickle.load(open(model_file_path, "rb"))
    data = pd.read_csv(test_data_path)
    X, y = preprocess_data(data)
    y_pred = lr_model.predict(X)
    return y_pred

# Function to get summary statistics


def dataframe_summary(data_path):
    """
    Summary statistics for the given data

    Args:
        data_path (str): data file path

    Returns:
        statistics_summary (list): statistics for the given data in list
    """
    # calculate summary statistics here
    logger.info("[Summary] Calculating summary statistics")
    data = pd.read_csv(data_path)
    X, y = preprocess_data(data)

    # calculate summary statistics of the training data
    summary = X.agg(['mean', 'median', 'std', 'min', 'max'])

    statistics_summary = [list(summary[col]) for col in X.columns]
    return statistics_summary


def missing_data_check(data_path):
    """
    Checking missing for the given data

    Args:
        data_path (str): data file path

    Returns:
        missing_values (list): missing percentage for the given data in list
    """
    data = pd.read_csv(data_path)
    X, y = preprocess_data(data)

    missing_values_df = X.isna().sum() / X.shape[0]
    return missing_values_df.values.tolist()

# Function to get timings


def execution_time():
    """
    Calculating the execution time

    Returns:
        ingestion_timing (float): total execution time for ingestion
        training_timing (float): total execution time for training
    """
    start_time = timeit.default_timer()
    os.system("python ingestion.py")
    ingestion_timing = timeit.default_timer() - start_time
    logger.info(
        f"[Time] Calculating execution time for ingestion: {ingestion_timing} seconds")

    start_time = timeit.default_timer()
    os.system("python training.py")
    training_timing = timeit.default_timer() - start_time
    logger.info(
        f"[Time] Calculating execution time for training: {training_timing} seconds")
    return ingestion_timing, training_timing

# Function to check dependencies


def outdated_packages_list():
    """
    Checking dependencies

    Returns:
        df (pd.DataFrame): dataframe of contains information of packages
    """
    logging.info("[Dependencies] Checking packages versions...")
    df = {
        "package_name": [],
        "current_version": [],
        "recent_version": []
    }

    with open("requirements.txt", "r") as file:
        strings = file.readlines()

        for line in strings:
            package_name, cur_ver = line.strip().split('==')
            df['package_name'].append(package_name)
            df['current_version'].append(cur_ver)
            info = subprocess.check_output(
                ['python', '-m', 'pip', 'show', package_name])
            df["recent_version"].append(str(info).split('\\n')[1].split()[1])

    df = pd.DataFrame(df)
    return df


if __name__ == '__main__':
    cfg = Config()
    model_file_path = os.path.join(cfg.output_model_path, "trainedmodel.pkl")
    test_data_path = os.path.join(cfg.test_data_path, "testdata.csv")
    model_predictions(model_file_path, test_data_path)
    dataframe_summary(test_data_path)
    missing_data_check(test_data_path)
    execution_time()
    outdated_packages_list()
