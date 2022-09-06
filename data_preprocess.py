import os
import pandas as pd
import numpy as np
from csv import reader
from collections import defaultdict
import os, sys, json
from sklearn.preprocessing import OneHotEncoder

def data_cleaning(): 
    """
    Clean data into useable format, preprocess data, drop unwanted columns
    """
    f = open('config.json')
    config = json.load(f)
    path = config['file_path']

    df = pd.read_csv(path,converters={"longitude": lambda x: x.strip("[]").split(", "), 
        "altitude": lambda x: x.strip("[]").split(", "), 
        "latitude": lambda x: x.strip("[]").split(", "), 
        "heart_rate": lambda x: x.strip("[]").split(", ")})

    df = df.drop(columns=['speed', 'hydration', 'Unnamed: 0'])

    # Calculate mean for longitude, altitude and latitude, heart rate
    for col in ["longitude", "altitude", "latitude", "heart_rate"]: 
        df[col] = pd.Series([[float(idx) for idx in x] for x in df[col]])
        df[col] = df[col].apply(np.mean)

    return df

def calculate_default(df): 
    """
    Calculate default values for user who doesn't enter any command arguments
    """

    # calculate column mean for continuous variables
    df_mean = df[['longitude', 'altitude', 'latitude', 'heart_rate', 'humidity',
       'wind_direction', 'temperature', 'wind_speed']].mean()

    # find the most comment value for categorical variables
    df_mode = df[['type', 'gender']].mode()

    # concat two series
    df_default = pd.concat([df_mean, df_mode.iloc[0,:]])

    return df_default

def data_processing(df, input_features): 
    """
    Feature transformation

    Parameters
    ----------
    path: str, path to the dataset
    """

    # Split df into features and labels
    df_y = df['sport']
    df_x = df.drop(labels='sport', axis=1).drop(labels='userId', axis=1).drop(labels='id', axis=1)
    df_x = pd.concat([df_x.reset_index(drop=True), input_features])
    # Categorical Data
    enc = OneHotEncoder(handle_unknown='ignore', drop='first')
    enc.fit(df_x[['gender', 'type']])
    df_cat = pd.DataFrame(enc.transform(df_x[['gender', 'type']]).toarray())
    df_x = pd.concat([df_x.drop(['type', 'gender'], axis=1).reset_index(drop=True), df_cat], axis=1)
    df_x.columns = df_x.columns.map(str)
    return df_x.iloc[:-1, :], df_y,  df_x.iloc[-1,:]

