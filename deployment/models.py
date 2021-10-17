from Resnet_model import Resnet_model 
from XGB_model import XGB_model 
from FTT_model import FTT_model 

'''
This file contains the methods that make up the inference pipeline.
'''

def prediction_FTT(X_DL):
    model = FTT_model.build()
    return model.predict(X_DL)

def prediction_resnet(X_DL):
    print(X_DL.shape)
    model = Resnet_model.build()
    return model.predict(X_DL)

def prediction_XGB(X_DL):
    model = XGB_model.build()
    return model.predict(X_DL)

