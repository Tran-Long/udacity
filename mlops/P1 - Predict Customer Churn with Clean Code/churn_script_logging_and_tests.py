"""
Project: Predict Customer Churn
Author: longth28
Date: 2023/11/06
"""

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

DATA_PATH = "./data/bank_data.csv"


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data(DATA_PATH)
        logging.info("TEST: Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("TEST: Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "TEST: Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        df_local = cls.import_data(DATA_PATH)
        perform_eda(df_local)
        assert os.path.exists(
            os.path.join(
                cls.EDA_PATH,
                'churn_histogram.png'))
        assert os.path.exists(
            os.path.join(
                cls.EDA_PATH,
                'customer_age_histogram.png'))
        assert os.path.exists(
            os.path.join(
                cls.EDA_PATH,
                'marital_status_histogram.png'))
        assert os.path.exists(
            os.path.join(
                cls.EDA_PATH,
                'total_trans_cts_histogram.png'))
        assert os.path.exists(os.path.join(cls.EDA_PATH, 'correlation.png'))
        logging.info("TEST: Testing perform_eda: SUCCESS")

    except AssertionError as err:
        logging.error(
            "Testing perform_eda: Missing eda figure results")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df = cls.import_data(DATA_PATH)
    cls.perform_eda(df)
    try:
        df_encoded = encoder_helper(df, cls.CAT_COLUMNS, cls.RESPONSE)
        assert df_encoded.shape[0] == df.shape[0]
        logging.info("TEST: Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("TEST: Testing encoder_helper: FAILED!")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df = cls.import_data(DATA_PATH)
    cls.perform_eda(df)

    df = cls.encoder_helper(
        df=df,
        category_lst=cls.CAT_COLUMNS,
        response=cls.RESPONSE)

    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df, cls.RESPONSE)
        assert x_train.shape[0] == 7088 and y_train.shape[0] == 7088 \
            and x_test.shape[0] == 3039 and y_test.shape[0] == 3039
        logging.info("TEST: Testing perform_feature_engineering: SUCCESS")

    except AssertionError as err:
        logging.error(
            "TEST: Testing perform_feature_engineering: Shape mismatch")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    df = cls.import_data(DATA_PATH)
    cls.perform_eda(df)

    df = cls.encoder_helper(
        df=df,
        category_lst=cls.CAT_COLUMNS,
        response=cls.RESPONSE)

    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        df, cls.RESPONSE)

    try:
        train_models(x_train, x_test, y_train, y_test)
        assert os.path.exists(
            os.path.join(
                cls.MODEL_PATH,
                "logistic_model.pkl"))
        assert os.path.exists(os.path.join(cls.MODEL_PATH, "rfc_model.pkl"))
        assert os.path.exists(
            os.path.join(
                cls.TRAINING_RESULT_PATH,
                "feature_importance.png"))
        assert os.path.exists(
            os.path.join(
                cls.TRAINING_RESULT_PATH,
                "logistic_regression_cls_report.png"))
        assert os.path.exists(
            os.path.join(
                cls.TRAINING_RESULT_PATH,
                "random_forest_cls_report.png"))
        assert os.path.exists(
            os.path.join(
                cls.TRAINING_RESULT_PATH,
                "roc_curve.png"))
        assert os.path.exists(
            os.path.join(
                cls.TRAINING_RESULT_PATH,
                "shap_values.png"))
        logging.info("TEST: Testing train_models: SUCCESS")

    except AssertionError as err:
        logging.error(
            "TEST: Testing train_models: Missing artifacts (weights/plots)")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
