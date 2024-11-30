# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This is the project to implement best production coding practices.

## Files and data description
```
Source code: Project_1

.
├── README.md
├── churn_library.py
├── churn_script_logging_and_tests.py
├── data
│   └── bank_data.csv
├── images
│   ├── eda
│   │   ├── churn_histogram.png
│   │   ├── correlation.png
│   │   ├── customer_age_histogram.png
│   │   ├── marital_status_counts.png
│   │   └── total_trans_cts_histogram.png
│   └── train
│       ├── feature_importance.png
│       ├── logistic_regression_cls_report.png
│       ├── random_forest_cls_report.png
│       ├── roc_curve.png
│       └── shap_values.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
└── requirements.txt
```

## Running Files
### Installation
```bash
pip install -r requirements.txt
```

### Running the whole process
```bash
python churn_library.py
```
Artifacts saved after running the whole process:

  - Best model weights: *./models*
  - Analysis figures: *./images*:
    - EDA results: *./images/eda*
    - Training results: *./images/train*
  - Log: *./logs/churn_library.log* 

### Running tests for functions
```bash
python churn_script_logging_and_tests.py
```
Artifacts saved after running the whole process:
The log is saved in *./logs/churn_library.log* 

## Code Quality
- **Style Guide** - Use `autopep8` via the command line commands below:

```bash
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
```


- **Style Checking and Error Spotting** - Check the pylint score using the command below.

```bash
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

