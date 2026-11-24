Predict Customer Churn
Project Predict Customer Churn of ML DevOps Engineer Nanodegree (Udacity)
Project Description
This project predicts whether a bank customer is likely to churn.

The original customer churn analysis contained in churn_notebook.ipynb has been refactored into reusable Python functions following software engineering best practices.

The machine learning pipeline performs the following steps:

Loads customer data from a CSV file.
Performs exploratory data analysis (EDA) and saves visualization results.
Creates a binary Churn response variable.
Encodes categorical customer features.
Splits the processed data into training and testing datasets.
Trains classification models.
Evaluates the models using classification reports and ROC curves.
Generates a feature importance plot.
Saves the trained models for later use.
The project uses two classification models:

Random Forest Classifier
Logistic Regression
Files and Data Description
Main Files
churn_library.py
Contains reusable functions for data import, exploratory data analysis, categorical feature encoding, feature engineering, model training, model evaluation, and saving model outputs.

churn_script_logging_and_tests.py
Contains tests for the main functions in churn_library.py. The script verifies that functions execute correctly and that expected datasets, images, and model files are created. Test successes and failures are written to a log file.

churn_notebook.ipynb
Contains the original customer churn analysis used as the basis for the refactored Python implementation.

requirements.txt
Contains the Python dependencies required to run the project.

Data
data/bank_data.csv
The dataset contains customer information used to determine whether a customer is likely to leave the bank.

The original response information is stored in Attrition_Flag. The pipeline creates a binary Churn variable:

0 = Existing Customer
1 = Attrited Customer
The model uses customer characteristics such as age, transaction activity, credit information, account activity, and encoded categorical features.

Categorical features include:

Gender
Education Level
Marital Status
Income Category
Card Category
Output Directories
After running the project, outputs are saved to:

EDA images → images/eda/
Model evaluation results → images/results/
Trained models → models/
Test logs → logs/churn_library.log
The required directories are created automatically if they do not already exist.

Running the Files
1. Install Dependencies
Install the required Python packages:

python -m pip install -r requirements.txt
2. Run the Pipeline
Run:

python churn_library.py
The script:

Creates the required output directories.
Imports data/bank_data.csv.
Performs exploratory data analysis.
Creates and encodes the churn-related features.
Splits the data into training and testing sets.
Trains the Random Forest and Logistic Regression models.
Generates model evaluation results.
Saves the trained models.
3. Run Tests and Logging
Run:

python churn_script_logging_and_tests.py
The test script validates the main functionality of churn_library.py, including:

Data importing
EDA output generation
Categorical feature encoding
Feature engineering
Model training
Creation of model and evaluation files
Successful tests are logged as INFO messages. Failures are logged as ERROR messages.

The log file is saved at:

logs/churn_library.log
Expected Outputs
Running the project should generate the following outputs.

Models
models/rfc_model.pkl
models/logistic_model.pkl
These files contain the trained Random Forest and Logistic Regression models.

EDA Images
Exploratory data analysis plots are stored in:

images/eda/
These include visualizations of customer churn, customer characteristics, transaction behavior, and correlations between numerical features.

Model Evaluation Images
Model evaluation output is stored in:

images/results/
This includes:

Random Forest classification report
Logistic Regression classification report
ROC curve
Random Forest feature importance plot
Logs
logs/churn_library.log
The log records successful test execution and any errors encountered while testing the project.

Code Quality
The project follows PEP 8 style guidelines and can be checked using Pylint:

python -m pylint churn_library.py
python -m pylint churn_script_logging_and_tests.py
The implementation also separates data processing, visualization, feature engineering, model training, testing, and logging into reusable functions.

Notes
Output directories are automatically created when required.
Model training uses a fixed random state where applicable to support reproducible results.
Categorical features are converted into numerical features before model training.
The test script checks for both non-empty function outputs and expected output files.
Model performance depends on the supplied bank_data.csv dataset and the selected features.