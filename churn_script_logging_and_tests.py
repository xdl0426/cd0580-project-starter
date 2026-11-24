"""
Test and logging module for the customer churn prediction project.

This module tests functions in churn_library.py and records successful
operations and errors in a log file.

Author: Donglin Xuan
Date created: November 20, 2026
"""

import logging
import os
import churn_library as cls

LOGS_DIR = "./logs"
LOG_FILE = os.path.join(LOGS_DIR, "churn_library.log")
DATA_PATH = "./data/bank_data.csv"

os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_import(import_data):
    """
    Test that the data import function returns a non-empty DataFrame.

    Inputs:
        import_data (callable): Function that reads the customer churn
            CSV file and returns a pandas DataFrame.

    Outputs:
        None
    """
    try:
        df = import_data(DATA_PATH)
        logging.info("Testing import_data: SUCCESS")

    except FileNotFoundError as err:
        logging.error(
            "Testing import_data: ERROR - file not found: %s",
            err
        )
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

        logging.info("Testing import_data dataframe: SUCCESS")

    except AssertionError as err:
        logging.error(
            "Testing import_data dataframe: ERROR - dataframe is empty"
        )
        raise err


def test_eda(perform_eda):
    """
    Test that the EDA function generates the expected output files.

    Inputs:
        perform_eda (callable): Function that performs exploratory
            data analysis and saves EDA plots.

    Outputs:
        None
    """
    try:
        df = cls.import_data(DATA_PATH)
        perform_eda(df)

        # Assert output files exist
        assert os.path.exists("./images/eda/churn_distribution.png")
        assert os.path.exists("./images/eda/customer_age_distribution.png")
        assert os.path.exists("./images/eda/marital_status_distribution.png")
        assert os.path.exists("./images/eda/total_trans_ct_distribution.png")
        assert os.path.exists("./images/eda/correlation_heatmap.png")

        logging.info("Testing perform_eda: SUCCESS")

    except Exception as err:
        logging.error(
            "Testing perform_eda: ERROR - %s",
            err
        )
        raise err


def test_encoder_helper(encoder_helper):
    """
    Test that categorical features are encoded successfully.

    Inputs:
        encoder_helper (callable): Function that encodes categorical
            columns using the response variable.

    Outputs:
        None
    """
    try:
        df = cls.import_data(DATA_PATH)

        # Create response column
        df["Churn"] = df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

        # Call encoder_helper
        df = encoder_helper(
            df,
            cls.CATEGORY_COLUMNS,
            "Churn"
        )

        # Assert encoded columns exist
        for category in cls.CATEGORY_COLUMNS:
            assert f"{category}_Churn" in df.columns

        logging.info("Testing encoder_helper: SUCCESS")

    except Exception as err:
        logging.error(
            "Testing encoder_helper: ERROR - %s",
            err
        )
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """
    Test that feature engineering creates non-empty train and test sets.

    Inputs:
        perform_feature_engineering (callable): Function that prepares
            features and splits data into training and testing sets.

    Outputs:
        None
    """
    try:
        df = cls.import_data(DATA_PATH)

        # Create response column
        df["Churn"] = df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

        df = cls.encoder_helper(
            df,
            cls.CATEGORY_COLUMNS,
            "Churn"
        )

        # Perform feature engineering
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df,
            "Churn"
        )

        # Assert outputs are not empty
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0

        logging.info("Testing perform_feature_engineering: SUCCESS")

    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering: ERROR - %s",
            err
        )
        raise err


def test_train_models(train_models):
    """
    Test that model training creates the expected models and result files.

    Inputs:
        train_models (callable): Function that trains and evaluates the
            classification models and saves their output artifacts.

    Outputs:
        None
    """
    try:
        df = cls.import_data(DATA_PATH)

        # Create response column
        df["Churn"] = df["Attrition_Flag"].apply(
            lambda val: 0 if val == "Existing Customer" else 1
        )

        df = cls.encoder_helper(
            df,
            cls.CATEGORY_COLUMNS,
            "Churn"
        )

        # Prepare train and test data
        x_train, x_test, y_train, y_test = (
            cls.perform_feature_engineering(df, "Churn")
        )

        # Train models
        train_models(
            x_train,
            x_test,
            y_train,
            y_test
        )

        # Assert model files exist
        assert os.path.exists("./models/rfc_model.pkl")
        assert os.path.exists("./models/logistic_model.pkl")

        # Assert result images exist
        assert os.path.exists(
            "./images/results/rf_classification_report.png"
        )
        assert os.path.exists(
            "./images/results/lr_classification_report.png"
        )
        assert os.path.exists(
            "./images/results/roc_curve.png"
        )
        assert os.path.exists(
            "./images/results/feature_importances.png"
        )

        logging.info("Testing train_models: SUCCESS")

    except Exception as err:
        logging.error(
            "Testing train_models: ERROR - %s",
            err
        )
        raise err


if __name__ == "__main__":

    # Ensure logs directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)

    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)

    print("Tests completed. Check logs for details.")
