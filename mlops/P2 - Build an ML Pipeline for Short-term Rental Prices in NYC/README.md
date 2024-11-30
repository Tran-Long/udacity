# Build an ML Pipeline for Short-term Rental Prices in NYC

[**Project Description**](#project-description) | [**Install**](#install) | [**Login to Wandb**](#login-to-wandb) | [**Step-by-step**](#step-by-step) | [**Public Wandb project**](#public-wandb-project)

## Project Description
Working for a property management company renting rooms and properties for short periods of time on various platforms. Need to estimate the typical price for a given property based on the price of similar properties. Your company receives new data in bulk every week. The model needs to be retrained with the same cadence, necessitating an end-to-end pipeline that can be reused.


```bash
Tran-Long/build-ml-pipeline-for-short-term-rental-prices
├── cookie-mlflow-step
├── components
├── images
├── src
├── MLproject
├── README.md
├── conda.yml
├── config.yaml
├── environment.yml
└── main.py

```

## Install
In order to run these components you need to have conda (Miniconda or Anaconda) and MLflow installed.
```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
```

## Login to Wandb
```bash
wandb login
```

## Step-by-step
### 0. Full pipeline
```bash
mlflow run .
```

### 1. Download data
```bash
mlflow run . -P steps=download
```

### 2. EDA
```bash
mlflow run src/eda
```

### 3. Basic cleaning
```bash
mlflow run . -P steps=basic_cleaning
```

### 4. Check data
```bash
mlflow run . -P steps=data_check
```

### 5. Split data
```bash
mlflow run . -P steps=data_split
```

### 6. Train and evaluate model
```bash
mlflow run . -P steps=train_random_forest
```

Optimize hyper-parameters
```bash
mlflow run . \
    -P steps=train_random_forest \
    -P hydra_options="modeling.random_forest.max_depth=10,50,100 modeling.random_forest.n_estimators=100,200,500 -m"
```

### 7. Test model
```bash
mlflow run . -P steps=test_regression_model
```

### 8. Test model with new dataset
```
mlflow run https://github.com/Tran-Long/build-ml-pipeline-for-short-term-rental-prices.git -v 1.0.1 -P hydra_options="etl.sample='sample2.csv'"

```

## Public Wandb project
Link: https://wandb.ai/parzival37/nyc_airbnb?workspace=user-parzival37
