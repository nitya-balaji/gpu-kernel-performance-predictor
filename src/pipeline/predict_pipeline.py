import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object 

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features): #load respective .pkl files
        try:
            model_path="artifacts/model.pkl"
            preprocessor_path="artifacts/preprocessor.pkl"
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            #scale data inputted by user
            data_scaled=preprocessor.trasnform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 MWG:float, NWG:float, KWG:float, MDIMC:float,
               NDIMC:float, MDIMA:float, NDIMB:float, KWI:float,
               VWM:float, VWN:float, STRM:float, STRN:float, SA:float, SB:float):
        self.MWG=MWG
        self.NWG = NWG
        self.KWG=KWG
        self.MDIMC=MDIMC
        self.NDIMC=NDIMC
        self.MDIMA=MDIMA
        self.NDIMB=NDIMB
        self.KWI=KWI
        self.VWM=VWM
        self.VWN=VWN
        self.STRM=STRM
        self.STRN=STRN
        self.SA=SA
        self.SB=SB
    
    #turning all inputs provided by user will be mapped to each variable to create a dataframe (data frame is the form our model is used to dealing with)
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "MWG": [self.MWG],
                "NWG": [self.NWG],
                "KWG": [self.KWG],
                "MDIMC":[self.MDIMC],
                "NDIMC":[self.NDIMC],
                "MDIMA":[self.MDIMA],
                "NDIMB":[self.NDIMB],
                "KWI":[self.KWI],
                "VWM":[self.VWM],
                "VWN":[self.VWN],
                "STRM":[self.STRM],
                "STRN":[self.STRN],
                "SA":[self.SA],
                "SB":[self.SB],
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
            
        