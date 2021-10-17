import pandas as pd
import numpy as np
import models

'''
This file contains the methods that make up the data processing pipeline.
'''

def classify(filePath):
    '''
    This method makes the predictions using the XGBoost, 
    Resnet and Transformer based models. 

    @param filePath: local file path to where the uploaded .csv file is stored

    @return [pred_XGB, pred_ResNet, pred_FTT] array of predictions by different models
    '''
    # get inputs
    X, X_DL = processData(filePath)

    # predictions by different models
    pred_XGB = models.prediction_XGB(X)
    pred_ResNet = models.prediction_resnet(X_DL)
    pred_FTT = models.prediction_FTT(X_DL)

    return pred_XGB, pred_ResNet, pred_FTT



def processData(filePath):
    '''
    This is a helper function called by classify(). 
    It takes the file path, and transforms the data according to 
    how the models were trained for prediction. 

    @param filePath: local file path to where the uploaded .csv file is stored

    @return [X, X_DL] arrays of inputs. The former is for a ML model, the latter for DL.
    '''
    # columns
    cat_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                    'Contract', 'PaperlessBilling', 'PaymentMethod', 'Default',
                    'HighMonthlyCharges', 'ShortTenure']
    cts_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # read in csv file
    df = pd.read_csv(filePath).drop('customerID', axis = 1)

    # convert categorical variables to numbers
    base = pd.read_csv('cleaned_default.csv', index_col = 0).drop(['Default','customerID'], axis = 1)
    df_base = pd.concat([df, base], axis = 0)
    df_base[cat_columns[0:16]] = df_base[cat_columns[0:16]].apply(lambda x : x.astype('category').cat.codes)
    
    # engineer features
    # high monthly charges
    median = df_base["MonthlyCharges"].median()
    df_base["HighMonthlyCharges"] = df.apply(lambda row : 1 if row.MonthlyCharges > median else 0, axis = 1)

    # short tenure
    median = df_base["tenure"].median()
    df_base["ShortTenure"] = df.apply(lambda row : 1 if row.tenure < median else 0, axis = 1)

    # restore
    df = df_base.iloc[0:1]

    # to numpy
    X = df.to_numpy()

    # encode and normalize, for DL models
    X_DL = encodeCat_normalizeCts(df, cat_columns, cts_columns)
    return X, X_DL



def encodeCat_normalizeCts(df, cat_columns, cts_columns):
    '''
    This is a helper function called by processData(). 
    It encodes categorical variables and normalizes continuous ones.

    @param cat_columns: string names of columns that are categorical
    @param cts_columns: string names of columns that are continuous

    @return X: an output array that has encoded and normalized variables
    '''
   
    # split data set into continuous and categorical, and encode/normalize
    df_model_cat = pd.DataFrame()
    df_model_cts = pd.DataFrame()

    # one hot encode categorical variables
    base = pd.read_csv('base.csv', index_col = 0).drop(['Default','customerID'], axis = 1)
    df_base = pd.concat([df, base], axis = 0)
    for col in cat_columns:
        if col != "customerID" and col != 'Default':
            temp = pd.get_dummies(df_base[col], prefix = col)
            temp = temp.iloc[0:1]
            df_model_cat = pd.concat([df_model_cat, temp], axis = 1)

    # append and standardize continuous variables
    for col in cts_columns:
        df_model_cts = pd.concat([df_model_cts, df[col]], axis = 1)

    X_cat = df_model_cat
    X_cts = df_model_cts
    print(X_cat.shape, X_cts.shape)
    
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler()
    X_cts = np.log(X_cts)
    X_cts = s.fit_transform(X_cts)
    X = np.concatenate([X_cat, X_cts], axis = 1)

    # concatenate standardized continuous and categorical
    return X

