"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn import preprocessing
import pickle
import json

# function that changes timestamp to 24 hour format and converts it to seconds
def to_datetime(input_df):
    times = []
    for col in input_df.columns.values:
        if col.endswith('Time'):
            times.append(col)
        else:
            pass
    input_df[times] = input_df[times].apply(lambda x: pd.to_datetime(x, format='%I:%M:%S %p') )
    for i in times:
        input_df[i] = input_df[i].dt.time
        input_df[i] = input_df[i].apply(lambda x: 3600 * int(str(x)[0:2]) + 60 * int(str(x)[3:5]) + int(str(x)[7:]))
            
    return input_df

    # function that takes care of missing values
def impute_nan(input_df):
        
    def imp_mean(d):
        return d.fillna(round(d.mean(),1))
        
    def imp_mode(d):
        A = [x for x in d if pd.notnull(x) == True]
        most = max(list(map(A.count, A)))
        m = sorted(list(set(filter(lambda x: A.count(x) == most, A))))
        return d.fillna(m[0])
        
    for col in input_df.columns.values:
        if is_numeric_dtype(input_df[col]) == True:
            input_df[col] = input_df[col].transform(imp_mean)
        else:
            input_df[col] = input_df[col].transform(imp_mode)
                
    return input_df

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.read_json(data)

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Preprocessing steps --------


    feature_vector_df = to_datetime(feature_vector_df)

    feature_vector_df.drop(columns = ['Vehicle Type', 'User Id', 'Precipitation in millimeters', 'Rider Id'], inplace = True)
    
    feature_vector_df = impute_nan(feature_vector_df)

    # get dummy variables for Platform Type
    personal_dumm = pd.get_dummies(feature_vector_df['Personal or Business'], drop_first=True)
    feature_vector_df = pd.concat([feature_vector_df, personal_dumm], axis=1)
    platf_dumm = pd.get_dummies(feature_vector_df['Platform Type'], prefix = 'Plat', drop_first = True)
    feature_vector_df = pd.concat([feature_vector_df, platf_dumm], axis=1)

    feature_vector_df = feature_vector_df.set_index("Order No")
    feature_vector_df.drop(columns=['Platform Type'], inplace=True)
    feature_vector_df.drop(columns=['Personal or Business'], inplace=True)

    predict_vector = feature_vector_df
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
