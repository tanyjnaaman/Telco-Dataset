from xgboost import XGBClassifier

class XGB_model:
    
    def __init__(self, model_file):
        self.model = XGBClassifier()
        import os
        root = os.getcwd()
        #os.chdir(root + '\development')
        self.model.load_model(model_file)
        #os.chdir(root + '\deployment')

    def build(model_file = 'XGB_model_v1.bin'):
        return XGB_model(model_file)

    def predict(self, X):    
        pred = self.model.predict(X)
        if pred[0] == 1:
            return 'Default'
        else: 
            return 'Not default'


t = XGB_model.build()
import numpy as np
test = np.array([[  1.  ,   0.  ,   1.  ,   0.  ,  10.  ,   1.  ,   0.  ,   1.  ,
          0.  ,   2.  ,   0.  ,   2.  ,   2.  ,   2.  ,   0.  ,   1.  ,
          2.  ,  99.85, 990.9 ,   1.  ,   1.  ]])
print(t.predict(test))

    
