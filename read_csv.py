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
    avg = df["rating"].mean()
    for i in range(len(df)):
        userID = int(df.loc[i][0]-1)
        movieID = int(df.loc[i][1]-1)
        rating = float(df.loc[i][2])
        ratings[userID, movieID] = rating - avg
    return ratings

def learning(learning_rate, epoch, ratings):
    V = np.zeros(ratings.shape[0])
    U = np.zeros(ratings.shape[1])
    for _ in range(epoch):
        F = np.add.outer(V,U)
        gradient_V = np.sum(ratings-F, axis = 1) - 2*config.lamda*V
        gradient_U = np.sum(ratings-F, axis = 0) - 2*config.lamda*U
        V += 2*learning_rate*gradient_V
        U += 2*learning_rate*gradient_U
    return V, U

if __name__ == "__main__":
    df = read(config.file_path)
    ratings = convert_to_matrix(df)
    users, movies = learning(config.learning_rate, config.epoch, ratings)
    bias = np.add.outer(users, movies)
    balanced_df = pd.DataFrame(ratings-bias)
    save(balanced_df)
