import logging
import sys
import os

# Get the root directory (project root)
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add root to sys.path
sys.path.append(root_dir)
from utils.Handytoolbox import *
import pandas as pd
class IngestData:
    
    """
    Ingesting the data from the data_path

    """

    def __init__(self, data_path: str):
        """
        Args:
        data_path: path to the data
    
        """
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    
def ingest_df(data_path: str)-> pd.DataFrame:
    '''Ingesting the data from the data_path.

    Args:
         data_path: path to the data
    Returns:
        pd.DataFrame: the ingested data
    '''
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data:{e}")
        raise e
    


    