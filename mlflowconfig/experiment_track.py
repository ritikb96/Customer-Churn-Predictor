import os
import sys
import mlflow
import mlflow.sklearn
import logging
# Get the root directory (project root)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add root to sys.path
sys.path.append(root_dir)
from utils.Handytoolbox import *
mlflow.set_experiment("Churn Prediction")
mlflow.set_tracking_uri("http://127.0.0.1:5000")



def log_model(model,report):
    model_name = model.__class__.__name__
    with mlflow.start_run(run_name= model_name):
        logging.info(model_name,report)

        mlflow.log_param('model',model_name)
        mlflow.log_metric('accuracy',report['accuracy'])
        mlflow.log_metric('recall_class_1', report['1']['recall'])
        mlflow.log_metric('recall_class_0', report['0']['recall'])
        mlflow.log_metric('f1_score_macro', report['macro avg']['f1-score'])   
        mlflow.sklearn.log_model(model,'model')
        logging.info("model logged succesfully")      
