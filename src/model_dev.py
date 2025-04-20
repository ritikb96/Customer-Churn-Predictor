import os
import sys
# Get the root directory (project root)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add root to sys.path
sys.path.append(root_dir)
from utils.Handytoolbox import *
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import logging

class Model(ABC):
    """
    Abstract class for all models
    """

    @abstractmethod
    def train(self,X_train,y_train):
        """
        Trains the model
        ARgs:
            x_train: Training data
            y_train: Training labels
        """
        pass


class LinearRegressionModel(Model):
     """
    Linear Regression model
    """
     def train(self,X_train,y_train,**kwargs):
        """
        trains the model
        """

        try:
            reg = LinearRegression()
            reg.fit(X_train,y_train)
            logging.info("Model training completed")
            return reg
        except Exception as e:
            logging.error("Error in training model:{}".format(e))



class XGBoostModel:
    """
    This class implements an XGBoost model for multi-class classification.
    """
    def train(self, X_train, y_train, **kwargs):
        try:
            # Adjust labels to 0-4 (XGBoost expects 0-based labels for classification)
            y_train_adjusted = y_train - 1
            xg = xgb.XGBClassifier(
                objective='multi:softmax',  # Multi-class classification
                num_class=5,  # 5 classes (1 to 5)
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                eval_metric='mlogloss'  # Multi-class log loss
            )
            xg.fit(X_train, y_train_adjusted)
            logging.info("XGBoost model was trained for multi-class classification")
            return xg
        except Exception as e:
            logging.error(f"Error in training model: {e}")
            raise


class RandomForestModel:
    def train(self,x_train,y_train,**kwargs):
        try:
            rfc = RandomForestClassifier(class_weight= 'balanced', max_depth= 10, min_samples_leaf =1, min_samples_split= 17, n_estimators =149)

            rfc.fit(x_train,y_train)
            logging.info("Model training done.")
            return rfc
        except Exception as e:
            logging.error(f"Error in training model:{e}")

