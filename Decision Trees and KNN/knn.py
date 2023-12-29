#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 23:15:51 2022

@author: sahasraiyer
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math
from statistics import mode
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def normalize(features, labels):
    labels = labels.to_numpy()
    scaler = MinMaxScaler()
    features_norm = scaler.fit_transform(features)
    return features_norm, labels

#Function for predicting the label of current test instance
def predict(X_train, y_train, X_test, k):
    predictions = []
    for i in range(len(X_test)):
        dist = np.empty(len(X_train))
        for j in range(len(X_train)):
            dist[j] = math.dist(X_train[j], X_test[i])
        dist = np.argsort(dist)
        dist = dist[:k]
        predictions.append(mode(y_train[dist]))
    return predictions 

#Creating dataframe for ease in plotting

def createdf(accuracy_dict):
    data = []
    for k, acc in accuracy_dict.items():
        if(isinstance(acc, list)):
            for v in acc:
                data.append([k, v])

    df = pd.DataFrame(data, columns = ['k_values', 'averages'])
    return df

    



if __name__ == "__main__":
    iris_df = pd.read_csv("iris.csv", header=None)
    iris_df.head()
    
    k_values = [k for k in range(1, 52, 2)]
    
    train_acc, test_acc = [], []
    training_acc, testing_acc = {}, {}
    for k in k_values:
        training_acc[k] = []
        testing_acc[k] = []
        for i in range(20):
            iris_df = shuffle(iris_df)
            X, y = iris_df.iloc[:, :-1], iris_df.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
            X_train, y_train = normalize(X_train, y_train)
            X_test, y_test = normalize(X_test, y_test)
            
            training_pred = predict(X_train, y_train, X_train, k)
            testing_pred = predict(X_train, y_train, X_test, k)
            train_num_correct = np.sum(training_pred == y_train)
            test_num_correct = np.sum(testing_pred == y_test)
            
            train_accuracy = float(train_num_correct) / len(y_train)
            training_acc[k].append(train_accuracy)
            test_accuracy = float(test_num_correct) / len(y_test)
            testing_acc[k].append(test_accuracy)
            
    training_df = createdf(training_acc)
    training_df.rename(columns={'averages': 'training_averages'}, inplace=True)
    testing_df = createdf(testing_acc)
    testing_df.rename(columns={'averages': 'testing_averages'}, inplace=True)
    
    
    plot = sns.catplot(x='k_values', y='training_averages', data = training_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Accuracy plot for training data")
    #plot.show()
    
    plot = sns.catplot(x='k_values', y='testing_averages', data = testing_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title="Accuracy plot for testing data")
    #plot.show()
