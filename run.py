import os
import pandas as pd
import numpy as np
from csv import reader
from collections import defaultdict
from google.colab import drive
from sklearn.neighbors import NearestNeighbors
from inspect import ArgInfo
import argparse
import os, sys, json
from KNN_Recommender import KnnRecommender
from data_preprocess import data_cleaning, calculate_default, data_processing 


def parse_args(df_default):      
    """
    Add argument parser.
    If argument is not provided, the default value is 
    - mean for continuous variables
    - mode for categorical variables
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--longitude', type=float, default=df_default['longitude'],
                        help='Average Longitude of All Exercise History')
    parser.add_argument('--altitude', type=float, default=df_default['altitude'],
                        help='Average Altitude of All Exercise History')
    parser.add_argument('--latitude', type=float, default=df_default['latitude'],
                        help='Average Latitude of All Exercise History')
    parser.add_argument('--heart_rate', type=float, default=df_default['heart_rate'],
                        help='Average Heart Rate of All Exercise History')
    parser.add_argument('--gender', nargs='?', default=df_default['gender'],
                        help='Gender')
    parser.add_argument('--type', type=int, default=df_default['type'],
                        help='Weather Type of a Given Day')
    parser.add_argument('--humidity', type=float, default=df_default['humidity'],
                        help='Humidity of a Given Day')
    parser.add_argument('--wind_direction', type=float, default=df_default['wind_direction'],
                        help='Wind Direction of a Given Day')
    parser.add_argument('--temperature', type=float, default=df_default['temperature'],
                        help='Temperature of a Given Day')
    parser.add_argument('--wind_speed', type=float, default=df_default['wind_speed'],
                        help='Wind Speed of a Given Day')
    parser.add_argument('--n_top', type=int, default=5,
                        help='Number of Recommendations to Generate')
    return vars(parser.parse_args())

def main(): 
    # parse input argument
    df = data_cleaning()
    df_default = calculate_default(df)


    args = parse_args(df_default)
    input_features = pd.Series(args.values(), index=args.keys())[:-1]
    input_features = input_features.to_frame().transpose()
    df_x, df_y, inputs = data_processing(df, input_features)
    inputs = inputs.array.reshape(1, -1)

    f = open('config.json')
    config = json.load(f)
    KNN = KnnRecommender(df_x, df_y, 
        config['metric'], config['algorithm'], config['n_neighbors'], 
        config['n_jobs'])

    rec = KNN.recommendation(inputs, args['n_top'])
    
if __name__ == '__main__':
    main()


    