import json
import subprocess
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


logistic = LogisticRegression(random_state=1)
decisiontree = DecisionTreeClassifier(random_state=1)
randomforest= RandomForestClassifier(random_state=1,max_depth=10,n_estimators=50)
xgb= XGBClassifier(n_estimators=50,max_depth=4)
extratree=ExtraTreesClassifier(random_state=1) 
gradientbooster = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)

models = [logistic,decisiontree,randomforest, xgb, extratree,gradientbooster]
model_names=["logistic_model","decisiontree_model","randomforest_model","xgb_model",
          "extratree_model","gradientbooster_model"]