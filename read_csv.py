import pandas as pd
import numpy as np
from config import config

def read(path):
    df = pd.read_csv(path)
    return df

def save(df):
    df.to_csv("./dataset/balanced.csv")
    return

def convert_to_matrix(df):
    nusers = config.nusers
    nmovies = config.nmovies
    ratings = np.zeros(shape=(nusers,nmovies))
    true_value = np.zeros(shape=(nusers,nmovies))
    avg = df["rating"].mean()
    for i in range(len(df)):
        userID = int(df.loc[i][0]-1)
        movieID = int(df.loc[i][1]-1)
        rating = float(df.loc[i][2])
        ratings[userID, movieID] = rating - avg
        true_value[userID, movieID] = 1
    return ratings, true_value

def learning(learning_rate, epoch, ratings, true_value):
    V = np.zeros(ratings.shape[0])
    U = np.zeros(ratings.shape[1])
    for _ in range(epoch):
        if _%500 == 0:
            print(f"{_//500}/10 has finished")
            print(gradient_V[:10])
            print(V[:10])
        F = np.add.outer(V,U)
        gradient_V = np.sum(ratings-F*true_value, axis = 1)*0.98
        gradient_U = np.sum(ratings-F*true_value, axis = 0)*0.98
        V += 2*learning_rate*gradient_V
        U += 2*learning_rate*gradient_U
    return V, U

if __name__ == "__main__":
    df = read(config.file_path)
    ratings, true_value = convert_to_matrix(df)
    users, movies = learning(config.learning_rate, config.epoch, ratings, true_value)
    bias = np.add.outer(users, movies)
    balanced_df = pd.DataFrame(ratings-bias)
    save(balanced_df)
