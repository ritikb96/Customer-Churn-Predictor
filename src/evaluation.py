import logging
from abc import ABC,abstractmethod
import numpy as np
import json
from sklearn.metrics import mean_squared_error,root_mean_squared_error,r2_score,classification_report

class Evaluation(ABC):
    """
    Abstract class for defining strategy for evaluation of our models
    """

    @abstractmethod
    def calculate_scores(self, y_true:np.ndarray,y_pred:np.ndarray):
        """
        Calculates the scores for the model
        """

class MSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared Error
    """
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true,y_pred)
            logging.info("Mse:{}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE:{}".format(e))
            raise e
        
class R2(Evaluation):
    """
    Evaluation Strategy that uses R2 score
    """
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("Calculating r2")
            r2 = r2_score(y_true,y_pred)
            logging.info("R2{}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating r2:{}".format(e))
            raise e
        
class RMSE(Evaluation):
    """
    Evaluation Strategy that uses Root Mean Squared error
    """
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("Calculating rMSE")
            rmse = root_mean_squared_error(y_true,y_pred)
            logging.info("rMse:{}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error in calculating rMSE:{}".format(e))
            raise e

class ClassificationReport(Evaluation):
    """
    Evaluation strategy that is used to generate a classification report

    """
    def calculate_scores(self, y_true:np.ndarray, y_pred:np.ndarray):
        try:
            logging.info("processing classification report")
            report = classification_report(y_true, y_pred, output_dict=True)
            return report
        except Exception as e:
            logging.error("Error in generating the classification report:{}".format(e))
            raise e