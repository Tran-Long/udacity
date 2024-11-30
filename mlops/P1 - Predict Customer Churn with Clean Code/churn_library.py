"""
Project: Predict Customer Churn
Author: longth28
Date: 2023/11/06
"""

# import libraries
import logging
import os
import joblib
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

EDA_PATH = "./images/eda"
os.makedirs(EDA_PATH, exist_ok=True)
TRAINING_RESULT_PATH = "./images/train"
os.makedirs(TRAINING_RESULT_PATH, exist_ok=True)
MODEL_PATH = "./models"
os.makedirs(MODEL_PATH, exist_ok=True)
LOG_PATH = "./logs"
os.makedirs(LOG_PATH, exist_ok=True)

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

RESPONSE = "Churn"


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv
    output:
        df: pandas dataframe
    '''
    try:
        df = pd.read_csv(pth)
        logging.info("SUCCESS: Read csv file")
        return df
    except FileNotFoundError as err:
        logging.error("ERROR: File not found: %s", err)
        raise err


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
        df: pandas dataframe

    output:
        None
    '''
    logging.info("-----START EDA-----")
    logging.info(f"SUCCESS: Shape of df {df.shape}")
    logging.info(f"SUCCESS: Null count \n{df.isnull().sum()}")
    logging.info(f"SUCCESS: Describe df \n{df.describe()}")

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(os.path.join(EDA_PATH, "churn_histogram.png"))

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(os.path.join(EDA_PATH, "customer_age_histogram.png"))

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join(EDA_PATH, "marital_status_histogram.png"))

    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(os.path.join(EDA_PATH, "total_trans_cts_histogram.png"))

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join(EDA_PATH, "correlation.png"))

    plt.close("all")

    logging.info("SUCCESS: Save plots to %s", EDA_PATH)
    logging.info("-----FINISH EDA-----")


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be used for
                    naming variables or index y column]

    output:
        df: pandas dataframe with new columns for
    '''
    logging.info("-----START ENCODE CATEGORY FEATURE-----")
    for cat in category_lst:
        cat_lst = []
        cat_group = df.groupby(cat).mean()[response]
        for val in df[cat]:
            cat_lst.append(cat_group.loc[val])
        df[f"{cat}_{response}"] = cat_lst
    logging.info(f"SUCCESS: Encode categorical features {category_lst}")
    logging.info("-----FINISH ENCODE CATEGORY FEATURE-----")
    return df


def perform_feature_engineering(df, response):
    '''
    input:
        df: pandas dataframe
        response: string of response name [optional argument that could be used for
                    naming variables or index y column]

    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''
    logging.info("-----START FEATURE ENGINEER-----")
    y = df[response]
    X = pd.DataFrame()
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    logging.info("-----FINISH FEATURE ENGINEER-----")
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    output:
        None
    '''
    plt.rc('figure', figsize=(7, 7))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(
        os.path.join(
            TRAINING_RESULT_PATH,
            "random_forest_cls_report.png"))
    plt.close("all")

    plt.rc('figure', figsize=(7, 7))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(
        os.path.join(
            TRAINING_RESULT_PATH,
            "logistic_regression_cls_report.png"))
    plt.close("all")
    logging.info(
        f"SUCCESS: Save classification reports to {TRAINING_RESULT_PATH}")


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
        model: model object containing feature_importances_
        X_data: pandas dataframe of X values
        output_pth: path to store the figure

    output:
        None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(os.path.join(output_pth, "feature_importance.png"))
    plt.close("all")
    logging.info(f"SUCCESS: Save feature importance plot to {output_pth}")


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        None
    '''
    logging.info("-----START TRAINING MODELS-----")

    # making random forest & logistic regression model
    rfc = RandomForestClassifier(random_state=42)
    logging.info("SUCCESS: Create random forest model")
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    logging.info("SUCCESS: Create logistic regression model")
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # training models
    logging.info("Training models...")
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)
    logging.info("SUCCESS: Train models completed")

    # get predictions from best models
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # save classification reports
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # save roc-curve plot
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    _ = plot_roc_curve(lrc, X_test, y_test, ax=axis, alpha=0.8)
    _ = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=axis,
        alpha=0.8)
    plt.savefig(os.path.join(TRAINING_RESULT_PATH, "roc_curve.png"))
    plt.close("all")
    logging.info("SUCCESS: Save roc-curves plot to %s", TRAINING_RESULT_PATH)

    # save feature importance plot
    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        TRAINING_RESULT_PATH)

    # save best model
    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(
            MODEL_PATH,
            'rfc_model.pkl'))
    joblib.dump(lrc, os.path.join(MODEL_PATH, 'logistic_model.pkl'))
    logging.info("SUCCESS: Save best models to %s", MODEL_PATH)

    # save shap values
    # plt.figure(figsize=(15, 8))
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(
        shap_values,
        X_test,
        plot_type="bar",
        show=False,
        plot_size=[
            25,
            8])
    plt.savefig(os.path.join(TRAINING_RESULT_PATH, "shap_values.png"))
    plt.close("all")
    logging.info("SUCCESS: Save SAHP values plot to %s", TRAINING_RESULT_PATH)

    logging.info("-----FINISH TRAINING MODELS-----")


if __name__ == "__main__":
    dataframe = import_data("data/bank_data.csv")
    perform_eda(dataframe)
    encoder_helper(dataframe, CAT_COLUMNS, RESPONSE)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        dataframe, RESPONSE)
    train_models(X_train, X_test, y_train, y_test)
