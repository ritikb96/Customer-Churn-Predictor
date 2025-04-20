import logging
from abc import ABC, abstractmethod
from typing import Union
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


class DataStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values, and converts the data type to float.
        """
        try:
            data = data.drop("customerID",axis=1)
            data.dropna(inplace=True)

            #replace churn with 1 or 0
            data["Churn"] = data["Churn"].replace({"Yes":1,"No":0})

            #lets change  total charges 
            data["TotalCharges"] = data["TotalCharges"].replace({" ": "0.0"})
            data["TotalCharges"] = data["TotalCharges"].astype(float)



            #identifying columns with object datatype
            object_columns = data.select_dtypes(include="object").columns
            encoders ={}

            for column in object_columns:
                label_encoder = LabelEncoder()
                data[column] = label_encoder.fit_transform(data[column])
                encoders[column] = label_encoder
            
            #save the encoder to a pkl file
            with open("encoders.pkl","wb") as f:
                pickle.dump(encoders,f)


            return data
        except Exception as e:
            logging.error(e)
            raise e
        
    def inference_cleaning(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the data obtained during inference.
        Applies the encoders saved during training.
        """
        try:
            import pickle
            # Drop unnecessary columns
            data = data.drop("customerID", axis=1)

            # Fix TotalCharges
            data["TotalCharges"] = data["TotalCharges"].replace({" ": "0.0"}).astype(float)

            data.dropna(inplace=True)

             # Load the saved encoders
            with open("../encoders.pkl", "rb") as f:
                encoders = pickle.load(f)

            # Encode using the fitted encoders
            for col in data.select_dtypes(include="object").columns:
                if col in encoders:
                    encoder = encoders[col]
                    if not set(data[col]).issubset(set(encoder.classes_)):
                        raise ValueError(f"Unseen labels in column '{col}': {set(data[col]) - set(encoder.classes_)}")
                    data[col] = encoder.transform(data[col])
                else:
                    raise ValueError(f"No encoder found for column '{col}'")

            return data

        except Exception as e:
            logging.error(e)
            raise e




 

class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            x = data.drop("Churn", axis=1)
            y = data["Churn"]
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )

            smote = SMOTE(random_state=42)
            x_train_smote, y_train_smote = smote.fit_resample(x_train,y_train)

            return x_train, x_test, y_train, y_test
        except Exception as e:
            logging.error(e)
            raise e


class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_data(self.df)