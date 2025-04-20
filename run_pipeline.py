import os
import sys
# Get the root directory (project root)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add root to sys.path
sys.path.append(root_dir)
from utils.Handytoolbox import *
import logging
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot
import joblib
from steps.ingestdata import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluate import evaluate_model
from mlflowconfig.experiment_track import log_model

logging.basicConfig(level=logging.INFO)


if __name__ =='__main__':
    df = ingest_df(data_path='data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    x_train,x_test,y_train,y_test = clean_df(df)
    model = train_model(x_train,x_test,y_train,y_test)
    print(model.__class__.__name__)
    report = evaluate_model(model,x_test,y_test)
    joblib.dump(model,'models/model.pkl')

    log_model(model,report)

    
