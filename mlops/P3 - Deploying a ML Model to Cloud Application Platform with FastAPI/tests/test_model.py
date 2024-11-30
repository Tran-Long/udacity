import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from ml import process_data, inference, compute_model_metrics
import pickle
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="module")
def data():
    return pd.read_csv("data/census.csv")


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]


def test_data_column_names(data):
    """Assert that enough columns are in data"""
    expected_columns = sorted([
        'age',
        'workclass',
        'fnlgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'salary'
    ])
    exist_colums = sorted(list(data.columns))
    assert expected_columns == exist_colums


def test_data_nan(data):
    """Assert that data has no nan values"""
    for col_name in data.columns:
        assert not data[col_name].isnull().any(
        ), f"Column {col_name} has null values"


def test_inference(data):
    """
    Assert that inference function returns correct
    amount of predictions with respect to the input
    """

    _, test_df = train_test_split(data, test_size=0.2)
    [encoder, lb, dt_model] = pickle.load(open("model/dt.pkl", "rb"))

    X_test, y_test, _, _ = process_data(
        X=test_df,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )
    preds = inference(dt_model, X_test)

    assert len(preds) == len(X_test)


def test_model_metrics(data):
    """
    Assert that output metrics are in the correct range
    """

    _, test_df = train_test_split(data, test_size=0.2)
    [encoder, lb, dt_model] = pickle.load(open("model/dt.pkl", "rb"))

    X_test, y_test, _, _ = process_data(
        X=test_df,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb
    )

    preds = inference(dt_model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert precision >= 0.0 and precision <= 1.0
    assert recall >= 0.0 and recall <= 1.0
    assert fbeta >= 0.0 and fbeta <= 1.0
