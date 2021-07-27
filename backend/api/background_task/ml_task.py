
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
from ..utils.machine_learning import ML_utils
from .. import celery, db
from ..models import Company, FeatureName, FeatureScore, MetadataPipeline,MetaDataScore

from ..utils.machine_learning.ML_utils import model_names, models
warnings.filterwarnings("ignore")


@celery.task()
def train_model(data, company_email):
    #read data
    train  = ML_utils.read_data(data)
    train_original = train.copy()
    train_original = ML_utils.feature_engineering(train_original)
    
    company = Company.get_by_email(company_email)
    # TO DO 
    # add features of training model to DB
    feature_store = ML_utils.feature_store(train_original)
    feats = list(feature_store.keys())
    for feat in feats:
        feature = FeatureName(name=feat, company_id=company.id)
        feature.parameters = [FeatureScore(name='std', statistics=feature_store[feat]['std']), 
                              FeatureScore(name='mean', statistics=feature_store[feat]['mean']),
                              FeatureScore(name='median', statistics=feature_store[feat]['50%'])]
    
        db.session.add(feature)
        db.session.commit()
    
    
    #split data
    # x_train,x_test,y_train,y_test = ML_utils.split_data(train_original)
    return "hello"



