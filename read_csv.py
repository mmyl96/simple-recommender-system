import pandas as pd
import numpy as np
from config import config

def read(path):
    df = pd.read_csv(path)
    return df

def bias_save(name, vector):
    df = pd.DataFrame({name:[i for i in range(1, len(vector)+1)],
                       "bias": vector})
    df.to_csv(f"./dataset/{name}.csv")
    return

def save(ratings):
    temp = {"userID":[],
            "movieID":[],
            "rating":[]}
    print("Start")
    for user in range(config.nusers):
        for movie in range(config.nmovies):
            if ratings[user][movie]:
                temp["userID"].append(user+1)
                temp["movieID"].append(movie+1)
                temp["rating"].append(ratings[user][movie])
    print("End")
    print("Length of ratings: ", len(temp["userID"]))
    df = pd.DataFrame(temp)
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
    for _ in range(1, epoch+1):
        F = np.add.outer(V,U)
        gradient_V = np.sum(ratings-F*true_value, axis = 1) - 2*config.lamda * V
        gradient_U = np.sum(ratings-F*true_value, axis = 0) - 2*config.lamda * U
        V += 2*learning_rate*gradient_V
        U += 2*learning_rate*gradient_U
        if _%100 == 0:
            print(f"{_//100}/{epoch//100} has finished")
            print("Current Loss is: ", np.sum((ratings-F*true_value)**2, axis = (0,1)))
    bias_save("userID", V)
    bias_save("movieID", U)
    return V, U

if __name__ == "__main__":
    df = read(config.file_path)
    ratings, true_value = convert_to_matrix(df)
    users, movies = learning(config.learning_rate, config.epoch, ratings, true_value)
    bias = np.add.outer(users, movies)
    bias = bias * true_value
    balanced_ratings = ratings-bias
    save(balanced_ratings)
