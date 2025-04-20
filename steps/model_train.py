
import logging
import pandas as pd
from src.model_dev import LinearRegressionModel,XGBoostModel,RandomForestModel
from sklearn.base import RegressorMixin
from src.config import ModelNameConfig




def train_model(
    x_train:pd.DataFrame,
    x_test:pd.DataFrame,
    y_train:pd.Series,
    y_test:pd.Series,    ) -> RegressorMixin:
    """
    Trains the model on the ingested data.
    
    Args:
        x_train: pd.DataFrame
        x_test: pd.DataFrame
        y_train: pd.Series
        y_test: pd.Series
    Returns:
        model: RegressorMixin
    """
    try:
        config = ModelNameConfig()
        logging.info("Model training starting...")
        model = None        
        if  config.modelName == "linear_regression":
            model = LinearRegressionModel()
            model = model.train(x_train,y_train)
            return model
        elif config.modelName =="xgboost":
            model = XGBoostModel()
            model = model.train(x_train,y_train)
            return model
        elif config.modelName =="random_forest":
            model = RandomForestModel()
            model = model.train(x_train,y_train)
            return model
        else:
            raise ValueError("Model {} not suppported".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model:{}".format(e))
        raise e