import json
import subprocess
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def read_data(data):
    try:
        read_file = pd.read_csv(data)
    except ValueError:
        read_file = pd.read_excel(data)
    return read_file


def feature_engineering(data):
    data["Dependents"].replace("3+", 3, inplace=True)
    data["Loan_Status"].replace("N", 0, inplace=True)
    data["Loan_Status"].replace("Y", 1, inplace=True)

    # handle missing values
    data["Gender"].fillna(data["Gender"].mode()[0], inplace=True)
    data["Married"].fillna(data["Married"].mode()[0], inplace=True)
    data["Dependents"].fillna(data["Dependents"].mode()[0], inplace=True)
    data["Self_Employed"].fillna(data["Self_Employed"].mode()[0], inplace=True)
    data["Credit_History"].fillna(data["Credit_History"].mode()[0], inplace=True)
    data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mode()[0], inplace=True)
    data["LoanAmount"].fillna(data["LoanAmount"].median(), inplace=True)

    # create new features
    data["TotalIncome"] = data["ApplicantIncome"] + data["CoapplicantIncome"]
    data["EMI"] = data["LoanAmount"] / data["Loan_Amount_Term"]
    data["Balance_Income"] = data["TotalIncome"] - data["EMI"]

    # scale data
    Columns = ["LoanAmount", "TotalIncome", "Balance_Income"]
    for column in Columns:
        data[f"{column}_log"] = np.log(data[column])

    # drop uwanted columns
    data = data.drop(
        [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "TotalIncome",
            "Loan_Amount_Term",
            "Loan_ID",
            "LoanAmount",
            "Balance_Income",
        ],
        axis=1,
    )

    # encode categorical variables
    data = pd.get_dummies(data, drop_first=True)
    return data


def feature_store(data):
    """This records the statistical distribution for independent variables
    A standard deviation is a statistic that measures the dispersion of a dataset relative to its mean
    We are going to map out some important features, it would inform our CT pipelines
    """

    store = data.describe()[
        ["Credit_History", "LoanAmount_log", "TotalIncome_log", "Balance_Income_log"]
    ].to_dict()
    # To access a statistical feature : store["Credit_History"]["mean"]
    return store


def split_data(data):
    dependent_vars = data["Loan_Status"]
    independent_vars = data.drop("Loan_Status", 1)
    x_train, x_test, y_train, y_test = train_test_split(
        independent_vars, dependent_vars, test_size=0.3, random_state=1
    )
    return x_train, x_test, y_train, y_test


def eval_metrics(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 roc_auc_score)

    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return precision, accuracy, recall, roc_auc





def log_metrics(precision, accuracy, recall, roc_auc):
    mlflow.log_metric("acc", accuracy)
    mlflow.log_metric("roc", roc_auc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)


def log_parameters(dict_paramters):
    for key, value in dict_paramters.items():
        mlflow.log_param(key, value)


def pickle_model(classifier, model_name):
    mlflow.sklearn.log_model(classifier, model_name)
    model_uuid = mlflow.active_run().info.run_uuid
    print("Model run: ", mlflow.active_run().info.run_uuid)


def train_model_gridsearch(model, paramters, X_train, y_train):
    clf = GridSearchCV(model, paramters, cv=5, verbose=0)
    best_model = clf.fit(X_train, y_train)
    return best_model


def trees_methods(model, tree_hyperparameters, X_train, y_train):

    """tree have similar paramters , other than repeating code, just create a function"""
    best_model = train_model_gridsearch(model, tree_hyperparameters, X_train, y_train)
    # select best parameters
    max_depth = best_model.best_estimator_.get_params()["max_depth"]
    #     n_estimators = best_model.best_estimator_.get_params()['n_estimators']
    random_state = best_model.best_estimator_.get_params()["random_state"]
    final_paramters = dict(random_state=random_state, max_depth=max_depth)

    return max_depth, random_state, final_paramters
