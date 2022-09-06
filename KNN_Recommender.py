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

class KnnRecommender:
    """
    Item-Based Collaborative Filtering Recommender System with
    KNN implmented by sklearn
    """
    def __init__(self, df_x, df_y, metric, algorithm, n_neighbors, n_jobs):
        """
        Define KNN Recommender System

        Parameters
        ----------
        df_x: pd.DataFrame, data frame for features
        df_y: pd.DataFrame, data frame for labels
        metric: str or callable, default='minkowski'
        algorithm: str, {'auto', 'ball_tree', 'kd_tree', 'brute'}
        n_neighbors: int, number of neighbors to use
        n_jobs: int, number of parallel jobs to run for neighbors search
        """
        self.df_x = df_x
        self.df_y = df_y
        self.model = NearestNeighbors(metric=metric, algorithm=algorithm, 
            n_neighbors=n_neighbors, n_jobs=n_jobs)

    def recommendation(self, input_features, n_rec):
        """
        make n recommendations for sports,
        if the same sports occurs multiple times, choose the minimum distance

        Parameters
        ----------
        input_features: pd.Series, the features entered by the user
        n_rec: int, number of sports recommendations
        """
        # Fit Model
        self.model.fit(self.df_x.to_numpy())

        # Get Recommendation
        distances, indices = self.model.kneighbors(input_features)
        df_pred = self.df_y.iloc[indices.flatten()].to_frame(name='sports')
        df_pred['distance'] = distances.flatten()
        df_pred = df_pred.groupby(['sports'])['distance'].min().to_frame().reset_index()
        df_pred = df_pred.sort_values("distance")
        for i in range(df_pred.shape[0]): 
          print('Sports Recommendation {0} of {1} with distance of {2}'.format(
            i+1, df_pred.iloc[i, 0], round(df_pred.iloc[i, 1], 5)
          ))

