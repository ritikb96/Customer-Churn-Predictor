import logging
import pandas as pd
from src.evaluation import RMSE,MSE,R2,ClassificationReport
from sklearn.base import RegressorMixin,ClassifierMixin
from typing import Tuple
from typing_extensions import Annotated



def evaluate_model(model:ClassifierMixin,
                   x_test:pd.DataFrame,
                   y_test:pd.DataFrame,) -> Tuple[
                       Annotated[float,"r2_score"],
                       Annotated[float,"rmse"]
                       
                       
                   ]:


    """
    Evaluates the model 
    """
    try:
        prediction = model.predict(x_test)
        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test,prediction)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test,prediction)


        rmse_class = RMSE()
        rmse = rmse_class.calculate_scores(y_test,prediction)

        classification_report = ClassificationReport()
        report = classification_report.calculate_scores(y_test,prediction)


        return report
    
    except Exception as e:
        logging.error("Error in evaluating the model:{}".format(e))
        raise e  
