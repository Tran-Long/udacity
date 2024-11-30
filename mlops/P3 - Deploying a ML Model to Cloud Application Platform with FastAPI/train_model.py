import logging
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from ml import process_data, compute_model_metrics, inference, train_model
import yaml

logging.basicConfig(
    filename="./logs/train_log.txt",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def compute_metrics_with_cat_slices(
    data: pd.DataFrame,
    config
):
    """Compute metrics for categorical slices of data. Uses metrics: [precision, recall, f1]"""
    logger.info("Calculating metrics on slices of categorical features in data")
    cat_features = config["data"]["cat_features"]
    label = config["data"]["label"]
    [encoder, lb, model] = pickle.load(
        open(config["model"]["saved_model_path"], "rb"))
    cat_slices_result_path = config["slices"]["cat_slices_result_path"]
    os.makedirs(os.path.dirname(cat_slices_result_path), exist_ok=True)
    result_dict = {
        "column": [],
        "cat_value": [],
        "precision": [],
        "recall": [],
        "f1": []
    }
    for feature in cat_features:
        for value in data[feature].unique():
            feat_val_df = data[data[feature] == value]
            X_test, y_test, encoder, lb = process_data(
                feat_val_df,
                categorical_features=cat_features,
                label=label,
                encoder=encoder,
                lb=lb,
                training=False
            )
            y_preds = inference(model=model, X=X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, y_preds)
            result_dict["column"].append(feature)
            result_dict["cat_value"].append(value)
            result_dict["precision"].append(precision)
            result_dict["recall"].append(recall)
            result_dict["f1"].append(fbeta)

    logger.info(f"Storing cat-slices results to {cat_slices_result_path}")
    pd.DataFrame(result_dict).to_csv(cat_slices_result_path, index=False)


def train(config):
    # Read data
    logger.info(f"Loading data from {config['data']['path']}")
    data = pd.read_csv(config["data"]["path"])

    # Split the data into training and testing sets
    logger.info(
        f"Splitting data into train/test with test ration: {config['model']['test_size']}")
    train, test = train_test_split(
        data, test_size=config["model"]["test_size"])
    # Preprocess train data
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=config["data"]["cat_features"],
        label=config["data"]["label"],
        training=True
    )

    # Train the model and save the results
    logger.info("Training decision tree model...")
    dt_model = train_model(
        X_train,
        y_train,
        **config["model"]["decision_tree"])

    logger.info("Training decision tree model DONE")
    logger.info(
        f"Save [encoder, lb, dt_model] to {config['model']['saved_model_path']}")
    os.makedirs(
        os.path.dirname(
            config["model"]["saved_model_path"]),
        exist_ok=True)
    with open(os.path.join(config["model"]["saved_model_path"]), "wb") as f:
        pickle.dump([encoder, lb, dt_model], f)

    # Preprocess the test data
    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=config["data"]["cat_features"],
        label=config["data"]["label"],
        encoder=encoder,
        lb=lb,
        training=False
    )
    logger.info("Inferencing model...")
    y_preds = inference(model=dt_model, X=X_test)

    # Calculate the metrics on the predictions
    precision, recall, fbeta = compute_model_metrics(y_test, y_preds)
    logger.info(f">>>Precision: {precision}")
    logger.info(f">>>Recall: {recall}")
    logger.info(f">>>Fbeta: {fbeta}")

    # Calculate the metrics on the categorical slices of test data
    compute_metrics_with_cat_slices(test, config)


if __name__ == '__main__':
    with open("./config.yaml", "r") as f:
        config = yaml.load(f, yaml.SafeLoader)
    train(config)
