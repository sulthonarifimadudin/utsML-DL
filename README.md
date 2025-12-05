# utsML-DL

Hands-On End-to-End Models (Classification, Regression & Clustering)

1. Repository Purpose
This repository contains my individual submission for the Machine Learning Midterm (UTS) with the theme:

“Hands-On End-to-End Models Machine Learning and Deep Learning”

The focus in this repository is on traditional / classical machine learning for:

Fraud detection (binary classification)
Song year prediction (regression)
Customer segmentation (clustering)
Each task is implemented as a complete end-to-end ML pipeline: from data loading, preprocessing, modeling, evaluation, up to interpretation.

2. Project Overview
2.1 Objectives
Build end-to-end machine learning pipelines for:
Fraud detection on online transactions
Regression to predict the release year of songs
Customer clustering based on credit card behavior
Practice:
Data cleaning & preprocessing
Handling missing values and outliers
Handling class imbalance (for fraud detection)
Feature engineering / feature selection
Training and evaluating several ML models
Doing basic hyperparameter tuning
Comparing model performance and interpreting the results
2.2 Implemented Tasks
Fraud Detection – Binary Classification

Predict probability that a transaction is fraudulent (isFraud = 1).
Work with transaction-level features such as amount, time, product code, card & address information, etc.
Output can be:
Model performance on validation data
Fraud probability predictions for test_transaction.csv (for potential submission file: TransactionID, isFraud).
Song Year Prediction – Regression

Predict the release year of a song from numeric audio features.
First column in the dataset is the target (year), and the rest are anonymous numeric features (feature_1, feature_2, ...).
Customer Clustering – Unsupervised Learning

Group customers based on spending & payment behavior.
Use features such as balance, purchases, cash advance, frequency of transactions, credit limit, payments, minimum payments, tenure, etc.
Interpret what each cluster represents (e.g., high spenders, revolvers, transactors, risky customers, etc.).
3. Datasets
All datasets are provided by the lecturer as part of the midterm.

3.1 Fraud Detection – Transaction Data
train_transaction.csv

Labeled transactions for training & evaluation.
Each row = one online transaction with many features.
Target column: isFraud
1 → fraudulent transaction
0 → non-fraudulent transaction
test_transaction.csv

Same feature columns as train_transaction.csv, but without isFraud.
Used as input to trained model to generate fraud probability predictions.
Typical output format:
TransactionID, isFraud
3.2 Regression – Song Year Prediction
midterm-regresi-dataset.csv
First value in each row = target label (e.g., 2001) → release year of the song.
Remaining values = numeric audio features (feature_1, feature_2, ...).
Features are derived from the audio signal (e.g., timbre, spectral characteristics), without human-friendly names.
3.3 Customer Clustering – Credit Card Dataset
clusteringmidterm.csv
Each row = one credit card customer.
Important columns (not exhaustive):
CUST_ID: unique customer ID
BALANCE, BALANCE_FREQUENCY
PURCHASES, ONEOFF_PURCHASES, INSTALLMENTS_PURCHASES
PURCHASES_FREQUENCY, ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY
CASH_ADVANCE, CASH_ADVANCE_FREQUENCY, CASH_ADVANCE_TRX
PURCHASES_TRX
CREDIT_LIMIT
PAYMENTS, MINIMUM_PAYMENTS
PRC_FULL_PAYMENT
TENURE (months using the card)
4. Project Structure
Note: adjust filenames if your actual notebooks / folders use different names.

midterm-machine-learning/
├── data/
│   ├── train_transaction.csv
│   ├── test_transaction.csv
│   ├── midterm-regresi-dataset.csv
│   └── clusteringmidterm.csv
├── notebooks/
│   ├── 01_fraud_detection_classification_ml.ipynb
│   ├── 02_song_year_regression_ml.ipynb
│   └── 03_customer_clustering_ml.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models_classification.py
│   ├── models_regression.py
│   └── models_clustering.py
├── requirements.txt
└── README.md
