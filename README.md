# Fitness-Recommendation-System

The project constructs a recommender system for sport exercise

## Description

The dataset is obtained from RecFit Dataset. 
The model takes the longitude, altitude, latitude, heart_rate, gender, weather type, humidity, wind_direction, temperature and wind_speed
to recommende the type of sports for users

## Getting Started

### Executing program

```
python run.py 
```
default input arguments are the mean and mode of the features

### File Structures

- `EDA_of_Health.ipynb`: Exploratory Data Analysis using pySpark for 253,020 data points
- `run.py`: runs the entire recommender system to produce sports recommendation
- `config.json`: constructs the parameters of the KNN model
- `KNN_Recommender.py`: constructs KNN model structure 
- `data_preprocess.py`: create functions to conduct data cleaning and feature transformation
- `spark.ipynb`: use pySpark to create a baseline Kmeans model to predict sports type

## Authors

Lehan Li ll3745@nyu.edu

Ruixuan Zhang rx.zhang2000@gmail.com


## Acknowledgments

* [dataset](https://sites.google.com/eng.ucsd.edu/fitrec-project/home)
* [Movie Recommender System](https://github.com/KevinLiao159/MyDataSciencePortfolio/tree/master/movie_recommender)
