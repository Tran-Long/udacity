# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
A Decision Tree model were trained.

* Model version: 1.0.0
* Model date: 22 December 2023

## Intended Use
The model is used to predict the income classes on census data based on various information. There are two income classes, which are >50K and <=50K, respectively (this is considered as a binary classification task).

## Data
The UCI Census Income Data Set was used for this project. It consists of 32561 instances. Further information on the dataset can be found at https://archive.ics.uci.edu/ml/datasets/census+income
### Training Data
For training 80% of the data were used (26561 instances). 
### Evaluation Data
For evaluation 20% of the data were used (6513 instances).

## Metrics
Three metrics were used for model evaluation (these value are measured on the test set):
* precision: 0.7199055861526357
* recall: 0.5783817951959545
* f1: 0.6414300736067297

## Ethical Considerations
The dataset only consists of public available data with highly aggregated census data. Therefore, no harmful can be heppened from unintended use of the data.

## Caveats and Recommendations
It would be meaningful to either perform an hyperparameter optimization or try other models (e.g. ensemble models) to improve the metric results.
