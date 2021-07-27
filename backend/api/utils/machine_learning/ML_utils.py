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

logistic = LogisticRegression(random_state=1)
decisiontree = DecisionTreeClassifier(random_state=1)
randomforest= RandomForestClassifier(random_state=1,max_depth=10,n_estimators=50)
xgb= XGBClassifier(n_estimators=50,max_depth=4)
extratree=ExtraTreesClassifier(random_state=1) 
gradientbooster = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)

models = [logistic,decisiontree,randomforest, xgb, extratree,gradientbooster]
model_names=["logistic_model","decisiontree_model","randomforest_model","xgb_model",
          "extratree_model","gradientbooster_model"]


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


def run_gridsearch_cv(list_of_models,model_names, X_train, X_test, y_train, y_true):
    
    #first create model dictionary
    model_dict = dict(zip(model_names, list_of_models))
    
    precision_scores =[]
    recall_scores =[]
    accuracy_scores =[]
    roc_scores =[]
    #run the model
    for model in models:
        
        classifier_model = model.fit(X_train,y_train)
        y_pred = classifier_model.predict(X_test)
        precision, accuracy, recall, roc_auc = eval_metrics(y_true, y_pred)
        #append metrics
        precision_scores.append(precision)
        recall_scores.append(recall)
        accuracy_scores.append(accuracy)
        roc_scores.append(roc_auc)
        
    dataframe_dict = { "models":model_names,'acc': accuracy_scores, "recall":recall_scores,
                      'precision': precision_scores, 'roc_auc':roc_scores} 
    df = pd.DataFrame(dataframe_dict).sort_values(by=['roc_auc', 'acc'], ascending=False)
    
    #first log all model scores
    file = df.to_csv('all_models_score_before_gridsearch.csv')
    

    #select the best models
    classifier_name = df["models"].iloc[0]
    
    
    #list all parameters for tuning
    penalty = ['l1', 'l2']
    C= np.logspace(0,4,10)
    random_state=[2,4,5,8]
    n_estimators=[40,50,100]
    max_depth=[1,4,10,15]
    learning_rate = np.logspace(0,4,10)
    
    tree_hyperparameters = dict(random_state=random_state,max_depth=max_depth,n_estimators=n_estimators)
    linear_hyperparameters = dict(C=C, penalty=penalty, random_state=random_state)
    ensemble_parameters = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                               max_depth=max_depth, random_state=random_state)
    
    if classifier_name == "logistic_model":
        
        #apply gridsearch
        best_model = train_model_gridsearch(LogisticRegression(), linear_hyperparameters,X_train,y_train)
        
        #select best parameters
        penalty = best_model.best_estimator_.get_params()['penalty']
        C = best_model.best_estimator_.get_params()['C']
        random_state = best_model.best_estimator_.get_params()['random_state']
        final_paramters = dict(penalty=penalty, C=C, random_state=random_state)
        
        #run the final model
        classifier = LogisticRegression(C=C, penalty=penalty, random_state=random_state)
        classifier_model = classifier.fit(X_train,y_train)
        y_pred = classifier_model.predict(X_test)
        precision, accuracy, recall, roc_auc = eval_metrics(y_true, y_pred)
        
        #log metrics  
        log_metrics(precision, accuracy, recall, roc_auc)
        #log parameters
        log_parameters(final_paramters)
        #pickle model
        pickle_model(classifier_model , classifier_name)
        
        
    elif classifier_name == "decisiontree_model":
        max_depth, random_state, final_paramters = trees_methods(model, tree_hyperparameters,
                                                                             X_train,y_train)
        #run the final model
        classifier = DecisionTreeClassifier(random_state=random_state,max_depth=max_depth)
        classifier_model = classifier.fit(X_train,y_train)
        y_pred = classifier_model.predict(X_test)
        precision, accuracy, recall, roc_auc = eval_metrics(y_true, y_pred)
        
        #log metrics  
        log_metrics(precision, accuracy, recall, roc_auc)
        #log parameters
        log_parameters(final_paramters)
        #pickle model
        pickle_model(classifier_model , classifier_name)
        
    
    elif classifier_name == "randomforest_model":
        max_depth, random_state, final_paramters = trees_methods(model, tree_hyperparameters,
                                                                             X_train,y_train)
        #run the final model
        classifier = RandomForestClassifier(random_state=random_state,max_depth=max_depth)
        classifier_model = classifier.fit(X_train,y_train)
        y_pred = classifier_model.predict(X_test)
        precision, accuracy, recall, roc_auc = eval_metrics(y_true, y_pred)
        #log metrics  
        log_metrics(precision, accuracy, recall, roc_auc)
        #log parameters
        log_parameters(final_paramters)
        #pickle model
        pickle_model(classifier_model , classifier_name)
        
    
    elif classifier_name == "gradientbooster_model":
        max_depth, random_state, final_paramters = trees_methods(model, tree_hyperparameters,
                                                                             X_train,y_train)
        #run the final model
        classifier = GradientBoostingClassifier(random_state=random_state,max_depth=max_depth)
        classifier_model = classifier.fit(X_train,y_train)
        y_pred = classifier_model.predict(X_test)
        precision, accuracy, recall, roc_auc = eval_metrics(y_true, y_pred)
        #log metrics  
        log_metrics(precision, accuracy, recall, roc_auc)
        #log parameters
        log_parameters(final_paramters)
        #pickle model
        pickle_model(classifier_model , classifier_name)
        
        
    elif classifier_name == "extratree_model":
        max_depth, random_state, final_paramters = trees_methods(model, tree_hyperparameters,
                                                                             X_train,y_train)
        #run the final model
        classifier = ExtraTreesClassifier(random_state=random_state,max_depth=max_depth)
        classifier_model = classifier.fit(X_train,y_train)
        y_pred = classifier_model.predict(X_test)
        precision, accuracy, recall, roc_auc = eval_metrics(y_true, y_pred)
        #log metrics  
        log_metrics(precision, accuracy, recall, roc_auc)
        #log parameters
        log_parameters(final_paramters)
        #pickle model
        pickle_model(classifier_model , classifier_name)
        
        
    else :
        best_model = train_model_gridsearch(XGBClassifier(), ensemble_parameters,X_train,y_train)
        
        #select best parameters
        n_estimators= best_model.best_estimator_.get_params()['n_estimators']
        learning_rate = best_model.best_estimator_.get_params()['learning_rate']
        max_depth = best_model.best_estimator_.get_params()['max_depth']
        random_state = best_model.best_estimator_.get_params()['random_state']
        final_paramters = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                               max_depth=max_depth, random_state=random_state)
        
        #pickle model
        pickle_model(classifier_model , classifier_name)
        
        
        #run the final model
        
        classifier = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                               max_depth=max_depth, random_state=random_state)
        classifier_model = classifier.fit(X_train,y_train)
        y_pred = classifier_model.predict(X_test)
        precision, accuracy, recall, roc_auc = eval_metrics(y_true, y_pred)
        
        #log metrics  
        log_metrics(precision, accuracy, recall, roc_auc)
        #log parameters
        log_parameters(final_paramters)
        #pickle model
        pickle_model(classifier_model , classifier_name)