import pandas as pd
import numpy as np
from time import time

from read_csv import read
from config import config
from quick_select import TopK

def convert_to_matrix(df):
    nusers = config.nusers
    nmovies = config.nmovies
    ratings = np.zeros(shape=(nusers,nmovies))
    for i in range(len(df)):
        userID = int(df.loc[i][0]-1)
        movieID = int(df.loc[i][1]-1)
        rating = float(df.loc[i][2])
        ratings[userID, movieID] = rating
    return ratings

def similarity(u_i, u_k):
    # Cosine Similarity
    return np.dot(u_i, u_k)/(np.sqrt(np.dot(u_i, u_i))*np.sqrt(np.dot(u_k, u_k)))

def pred_rating(userID, movieID):
    values = np.zeros(config.nusers)
    u_idx = np.nonzero(ratings[:,movieID-1])[0] # Users who saw movie
    for u in u_idx:
        if u == userID:
            continue
        values[u] = similarity(ratings[userID-1], ratings[u])
    Q = TopK(values, config.MaxUserNumber) # Fast get results and their indices by using quick select
    sim, idx = Q.answer()
    if sum(abs(sim)) == 0: # Avoid zero division
        return 0
    ## Note that we do not only consider most similar but also most dissimilar
    return V[userID-1] + U[movieID-1] + sum([sim[i]*ratings[idx[i],movieID-1] for i in range(config.MaxUserNumber)])/sum(abs(sim))

def prediction(userID):
    m_idx = np.where(ratings[userID-1]==0)[0] # Movies which user did not see yet
    values = np.zeros(config.nmovies)
    for m in m_idx:
        values[m] = pred_rating(userID, m+1)
    return np.argmax(values)+1 # Return index of movieID with highest score

if __name__ == "__main__":
    ## Read preprocessed data
    df = read("./dataset/balanced.csv")
    df = df.drop(columns = ['Unnamed: 0'])
    ## Preprocess of Data
    ratings = convert_to_matrix(df)
    user = read("./dataset/userID.csv")
    movie = read("./dataset/movieID.csv")
    V = user[["bias"]].to_numpy()
    U = movie[["bias"]].to_numpy()
    user = int(input("Please input userID: "))
    start = time()
    ## Predicted movie of a particular user
    movie = prediction(user)
    print("Time cost:", (time()-start)/60)
    print("After computing, the most recommended movie is:", movie)
