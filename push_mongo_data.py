import os
import sys
import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

import certifi
ca=certifi.where()

import pandas as pd
import numpy as np
import pymongo
from ApiStock.exception.exception import ApiPredictionException
from ApiStock.logging.logger import logging

class NetworkDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise ApiPredictionException(e,sys)
        
    def csv_to_json_convertor(self,file_path):
        try:
            data=pd.read_csv(file_path)
            data.reset_index(drop=True,inplace=True)
            records=list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise ApiPredictionException(e,sys)
        
    def insert_data_mongodb(self,records,database,collection):
        try:
            self.database=database
            self.collection=collection
            self.records=records

            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            
            self.collection=self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise ApiPredictionException(e,sys)

    def get_latest_prediction_folder(self):
        try:
            predictions_dir = "inferences"
            folders = [f for f in os.listdir(predictions_dir) if os.path.isdir(os.path.join(predictions_dir, f))]
            latest_folder = max(folders, key=lambda x: os.path.getctime(os.path.join(predictions_dir, x)))
            return latest_folder
        except Exception as e:
            raise ApiPredictionException(e,sys)
        
if __name__=='__main__':
    networkobj = NetworkDataExtract()
    latest_folder = networkobj.get_latest_prediction_folder()
    FILE_PATH = f"inferences/{latest_folder}/all_last_points.csv"
    DATABASE = "BACHV_BB_STOCKS"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    Collection = f"Net_Data_{current_time}"
    
    records = networkobj.csv_to_json_convertor(file_path=FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_mongodb(records, DATABASE, Collection)
    print(no_of_records)
        


