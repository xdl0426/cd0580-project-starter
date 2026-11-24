"""
Customer churn prediction library.

This module contains functions for importing data, performing exploratory
data analysis, feature engineering, model training, and model evaluation.

Author: Donglin Xuan
Date created: November 20, 2026
"""

import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

os.environ["QT_QPA_PLATFORM"] = "offscreen"

EDA_DIR = "./images/eda"
RESULTS_DIR = "./images/results"
MODELS_DIR = "./models"
DATA_PATH = "./data/bank_data.csv"
LOGS_DIR = "./logs"
LOG_FILE = os.path.join(LOGS_DIR, "churn_library.log")

CATEGORY_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)


def create_output_directories():
    """
    Create output directories used by the project.

    input:
            None
    output:
            None
    """
    os.makedirs(EDA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def import_data(pth):
    """
    Return a dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    logging.info("Importing data from %s", pth)

    df = pd.read_csv(pth)

    logging.info(
        "Data imported successfully with %d rows and %d columns.",
        df.shape[0],
        df.shape[1]
    )

    return df


def perform_eda(df):
    """
    Perform EDA on df and save figures.

    input:
            df: pandas dataframe
    output:
            None
    """
    logging.info("Starting exploratory data analysis.")

    create_output_directories()

    # Create churn column
    if "Churn" not in df.columns:
        df["Churn"] = df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

    # Churn distribution
    plt.figure(figsize=(20, 10))
    df["Churn"].hist()
    plt.savefig(os.path.join(EDA_DIR, "churn_distribution.png"))
    plt.close()

    # Customer age distribution
    plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    plt.savefig(os.path.join(EDA_DIR, "customer_age_distribution.png"))
    plt.close()

    # Marital status distribution
    plt.figure(figsize=(20, 10))
    df["Marital_Status"].value_counts("normalize").plot(kind="bar")
    plt.savefig(os.path.join(EDA_DIR, "marital_status_distribution.png"))
    plt.close()

    # Total transaction count distribution
    plt.figure(figsize=(20, 10))
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    plt.savefig(os.path.join(EDA_DIR, "total_trans_ct_distribution.png"))
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(20, 10))
    numeric_df = df.select_dtypes(include=np.number)
    sns.heatmap(numeric_df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"))
    plt.close()

    logging.info("EDA figures saved successfully to %s.", EDA_DIR)


def encoder_helper(df, category_lst, response):
    """
    Encode categorical features.

    input:
            df: pandas dataframe
            category_lst: list of categorical columns
            response: response column name
    output:
            df: updated dataframe
    """

    logging.info("Starting categorical feature encoding.")

    for category in category_lst:
        category_groups = df.groupby(category)[response].mean()
        df[category + "_" + response] = df[category].map(category_groups)

    logging.info("Categorical feature encoding completed successfully.")

    return df


def perform_feature_engineering(df, response):
    """
    Split dataset into train and test sets.

    input:
              df: pandas dataframe
              response: response column name
    output:
              x_train, x_test, y_train, y_test
    """
    logging.info("Starting feature engineering.")

    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]

    x_data = df[keep_cols]
    y_data = df[response]

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.3,
        random_state=42
    )

    logging.info(
        "Feature engineering completed. Training rows: %d, testing rows: %d.",
        x_train.shape[0],
        x_test.shape[0]
    )

    return x_train, x_test, y_train, y_test

def feature_importance_plot(model, x_data, output_pth):
    """
    Save a feature importance plot for the trained model.

    input:
            model: trained model with feature_importances_ attribute
            x_data: dataframe containing model features
            output_pth: path where the plot will be saved
    output:
            None
    """
    logging.info("Generating feature importance plot.")

    importances = model.feature_importances_

    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    plt.bar(
        range(x_data.shape[1]),
        importances[indices]
    )

    plt.xticks(
        range(x_data.shape[1]),
        names,
        rotation=90
    )

    plt.savefig(output_pth)
    plt.close()

    logging.info(
        "Feature importance plot saved successfully to %s.",
        output_pth
    )


def classification_report_image(
    y_train,
    y_test,
    predictions
):
    """
    Generate and save classification reports for both trained models.

    input:
            y_train: true response values for the training set
            y_test: true response values for the testing set
            predictions: Logistic Regression training predictions

    output:
            None
    """
    logging.info("Generating classification report images.")

    create_output_directories()

    # Random Forest classification reports
    plt.figure(figsize=(10, 7))

    plt.text(
        0.01,
        1.0,
        "Random Forest Training Results",
        fontsize=10,
        fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.55,
        classification_report(y_train, predictions["train_rf"]),
        fontsize=10,
        fontproperties="monospace"
    )

    plt.text(
        0.01,
        0.45,
        "Random Forest Testing Results",
        fontsize=10,
        fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.0,
        classification_report(y_test, predictions["test_rf"]),
        fontsize=10,
        fontproperties="monospace"
    )

    plt.axis("off")
    plt.savefig(
        os.path.join(RESULTS_DIR, "rf_classification_report.png"),
        bbox_inches="tight"
    )
    plt.close()

    # Logistic Regression classification reports
    plt.figure(figsize=(10, 7))

    plt.text(
        0.01,
        1.0,
        "Logistic Regression Training Results",
        fontsize=10,
        fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.55,
        classification_report(y_train, predictions["train_lr"]),
        fontsize=10,
        fontproperties="monospace"
    )

    plt.text(
        0.01,
        0.45,
        "Logistic Regression Testing Results",
        fontsize=10,
        fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.0,
        classification_report(y_test, predictions["test_lr"]),
        fontsize=10,
        fontproperties="monospace"
    )

    plt.axis("off")
    plt.savefig(
        os.path.join(RESULTS_DIR, "lr_classification_report.png"),
        bbox_inches="tight"
    )
    plt.close()

    logging.info(
        "Classification report images saved successfully to %s.",
        RESULTS_DIR
    )


def train_models(x_train, x_test, y_train, y_test):
    """
    Train classification models, evaluate them, and save model artifacts.

    input:
            x_train: training feature data
            x_test: testing feature data
            y_train: training response data
            y_test: testing response data
    output:
            None
    """
    logging.info("Starting model training.")

    create_output_directories()

    # Create models
    rfc_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # Train models
    logging.info("Training Random Forest model.")
    rfc_model.fit(x_train, y_train)

    logging.info("Training Logistic Regression model.")
    lr_model.fit(x_train, y_train)

    # Make predictions
    predictions = {
        "train_rf": rfc_model.predict(x_train),
        "test_rf": rfc_model.predict(x_test),
        "train_lr": lr_model.predict(x_train),
        "test_lr": lr_model.predict(x_test),
    }

    logging.info("Model predictions generated successfully.")

    # Save classification reports
    classification_report_image(
        y_train,
        y_test,
        predictions
    )

    # Save ROC curve
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    RocCurveDisplay.from_estimator(
        rfc_model,
        x_test,
        y_test,
        ax=ax,
    )

    RocCurveDisplay.from_estimator(
        lr_model,
        x_test,
        y_test,
        ax=ax,
    )

    plt.savefig(
        os.path.join(RESULTS_DIR, "roc_curve.png")
    )
    plt.close()

    logging.info(
        "ROC curve saved successfully to %s.",
        RESULTS_DIR
    )

    # Save feature importance plot
    feature_importance_plot(
        rfc_model,
        x_train,
        os.path.join(RESULTS_DIR, "feature_importances.png")
    )

    # Save trained models
    joblib.dump(
        rfc_model,
        os.path.join(MODELS_DIR, "rfc_model.pkl")
    )

    joblib.dump(
        lr_model,
        os.path.join(MODELS_DIR, "logistic_model.pkl")
    )

    logging.info(
        "Trained models saved successfully to %s.",
        MODELS_DIR
    )
    logging.info("Model training completed successfully.")


if __name__ == "__main__":
    create_output_directories()

    churn_df = import_data(DATA_PATH)

    perform_eda(churn_df)

    churn_df = encoder_helper(churn_df, CATEGORY_COLUMNS, "Churn")
    x_train_main, x_test_main, y_train_main, y_test_main = (
        perform_feature_engineering(churn_df, "Churn")
    )

    train_models(
        x_train_main,
        x_test_main,
        y_train_main,
        y_test_main
    )
