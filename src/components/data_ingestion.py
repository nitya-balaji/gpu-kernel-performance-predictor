import os
import sys 
from src.exception import CustomException
from src.logger import logging 
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass #decorator (automatically writes __init__ method)
class DataIngestionConfig:
    #below listed paths/inputs that are needed (e.g. for where we want the output files to go)
    train_data_path: str=os.path.join("artifacts", "train.csv")
    test_data_path: str=os.path.join("artifacts", "test.csv")
    raw_data_path: str=os.path.join("artifacts", "raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() #this class now knows where the artifacts folder should be
    
    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion Method or Component")
        try:
            df=pd.read_csv("notebook/data/sgemm_product_v2.csv")
            logging.info("Read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) #looks at artifacts/train.csv and extracts just the folder name (artifacts) -> exist_ok = True means if folder already exists don't crash
            
            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True) #backup of the file we read
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) #does the split
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) #save train.csv and test.csv into artifacts folder
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of data is completed")
            return (
              #this sends the paths back to whoever called the method so they know where to find the split data for the next step (Data Transformation)  
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()
    data_transformation = DataTransformation() #start next step in pipeline after data ingestion complete
    train_arr, test_arr, _=data_transformation.initiate_data_transformation(train_data, test_data) #_ to ignore file path to preprocessor .pkl file
    model_trainer=ModelTrainer() #start next step in pipeline after data transformation complete
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))