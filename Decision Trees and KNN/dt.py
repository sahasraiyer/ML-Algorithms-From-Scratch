#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 22:58:37 2022

@author: sahasraiyer
"""

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from math import log2
import seaborn as sns
import matplotlib.pyplot as plt

class leaf:
    def __init__(self, data):
        count = {}
        for row in data:
            label = row[-1]
            if label not in count:
                count[label] = 0
            count[label]+=1
        self.predictions = count
    
class decision_node:
    def __init__(self, feature, value, true_branch, false_branch):
        self.feature = feature
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch
        

#Checks for condition row[feature]>=value

def query(row, feature, value):
    com_val = row[feature]
    if com_val >= value:
        return True
    else:
        return False
    
#Partitions the data into left and right based on who satisfies row[feature]>=value

def partition(data, feature, value):
    true_r, false_r = [],[]
    
    for index,row in enumerate(data):
        if query(row, feature, value):
            true_r.append(row)
        else:
            false_r.append(row)

    return true_r, false_r

#Gini criterion

def gini(data):
    count = {}
    for row in data:
        label = row[-1]
        if label not in count:
            count[label] = 0
        count[label]+=1
    impurity = 1
    for label in count:
        prob = count[label]/float(len(data))
        impurity -= prob**2
    return impurity


#Entropy calculation

def entropy(data):
    count = defaultdict(list)

    for row in data:
        label = row[-1]
        if label not in count:
            count[label] = 0
        count[label]+=1
    #print(count)
    total = float(len(data))
    class0 = count[0]/float(len(data)) if count[0] else 1
    class1 = count[1]/float(len(data)) if count[1] else 1
    #if class0 > 0 and class1 > 0:
    entropy = -(class0 * log2(class0) + class1 * log2(class1))
        
    return entropy


#Information gain, provide split type - gini or entropy

def infogain(left, right, uncertain, split_type):
    p = float(len(left))/(len(left)+len(right))
    if split_type == 'gini':
        info_gain = uncertain - p*gini(left)  - (1-p)*gini(right)
    elif split_type == 'entropy':
        info_gain = uncertain - p*entropy(left)  - (1-p)*entropy(right)
    return info_gain

# Run for all features and value queries, find best split

def find_best_split(data,split_type):
    best_gain = 0
    best_feature = None
    best_value = None
    uncertain = gini(data)
    #Iterates through all features
    all_features = len(data[0])-1
    for feature in range(all_features):
        #Finds the set of unique values which could be possible thresholds for each feature
        unique_values = set([row[feature] for row in data])
        #Check every possible threshold for split, if information gain on this split is highest, store as best feature to
        #carry split on
        for value in unique_values:
            true_rows, false_rows = partition(data, feature, value)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            infog = infogain(true_rows, false_rows, uncertain,split_type)
            
            if infog > best_gain:
                best_gain = infog
                best_feature = feature
                best_value = value
            
    return best_gain, best_feature, best_value

#Build tree from data, provide split type

def build(data,split_type = 'gini'):
    
    gain, feature, value = find_best_split(data,split_type)
    #If gain = 0, we have reached a node having 0 impurity, i.e, all instances in this split belong to the same class.
    #Hence, this node is considered to be a leaf node. 
    if gain == 0:
        return leaf(data)
    #If gain!=0, continue to partiton on feature, and build its corresponding left and right subtrees
    true_rows, false_rows = partition(data, feature, value)
    
    true_branch = build(true_rows)
    false_branch = build(false_rows)
    return decision_node(feature, value, true_branch, false_branch)


#Classification function

def classify(row, root):
    if isinstance(root, leaf):
        return root.predictions
    if query(row, root.feature, root.value):
        return classify(row, root.true_branch)
    else:
        return classify(row, root.false_branch)
    
    
#Stores the label, and the corresponding probability of predicted label

def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

#Accuracy function

def get_accuracy(test_data):
    correct = 0
    total = len(test_data)
    for i,row in enumerate(test_data):

        actual = row[-1]
        predicted = print_leaf(classify(row, root_node))
        if actual == list(predicted.keys())[0]:
            correct+=1
    accuracy = float(correct)/float(total)
    return accuracy

#Function to normalize the features

def normalize(features, labels):
    labels = labels.to_numpy()
    scaler = MinMaxScaler()
    features_norm = scaler.fit_transform(features)
    return features_norm.tolist(), labels.tolist()

def convert_tolist(X_train, X_test, y_train, y_test):
    X_train = X_train.values.tolist()
    X_test = X_test.values.tolist()
    y_train = y_train.values.tolist()
    y_test = y_test.values.tolist()
    
    return X_train, X_test, y_train, y_test

def create_dataset(data, data_label):
    full_data = []
    for row,label in zip(data, data_label):
        full_row = row
        full_row.append(label)
        full_data.append(full_row)
    return full_data


#The present implementation iteratively builds the decision tree up until the point a particular split has 0 entropy
# i.e. all the instances in that split belong to the same class. 

#To ensure a more robust model, that is less prone to overfit, a possible implementation will be to set a 
#threshold on the majority value of instances belonging to a class, on a split. 
#Once this split has been reached, the decsion tree willstop building any further below that node. 
#This will also ensure better generalization capabilities of the tree for unseen data


if __name__ == "__main__":
    house_data = pd.read_csv('house_votes_84.csv')
    house_data.head()
    
    #Using information gain as criterion

    train_acc, test_acc = [], []
    for i in range(100):
        house_data = shuffle(house_data)
        X = house_data.drop(['target'],axis=1)
        y = house_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        X_train, y_train = normalize(X_train, y_train)
        X_test, y_test = normalize(X_test, y_test)
    
        train_data = create_dataset(X_train, y_train)
        test_data = create_dataset(X_test, y_test)
        root_node = build(train_data,split_type='entropy')
        train_acc.append(get_accuracy(train_data))
        test_acc.append(get_accuracy(test_data))
        
    plt.hist(train_acc, bins = 30)
    plt.axvline(np.average(train_acc), color='black', linestyle='dashed', linewidth=2)
    plt.title("Accuracy histogram using Information Gain as criterion for Training Data")
    plt.xlabel("Accuracy")
    plt.ylabel("Accuracy Frequency on Training Data")
    plt.show()
    
    std_mean_train = np.std(train_acc) / np.sqrt(np.size(train_acc))
    print("Train accuracy mean : ", np.average(train_acc))
    print("Train accuracy mean standard deviation : ",std_mean_train)
    
    #import seaborn as sns
    plt.hist(test_acc)
    mean_accuracy = np.average(test_acc)
    plt.axvline(x=mean_accuracy,color='blue', ls = '--')
    plt.xlabel("Accuracy")
    plt.ylabel("Accuracy Frequency on Test Data")
    plt.title("Accuracy histogram using Information Gain as criterion for Testing Data")
    plt.show()
    
    std_mean = np.std(test_acc) / np.sqrt(np.size(test_acc))
    print("Test accuracy mean : ", mean_accuracy)
    print("Test accuracy mean standard deviation : ",std_mean)
    
    train_acc_gini, test_acc_gini = [], []
    for i in range(100):
        house_data = shuffle(house_data)
        X = house_data.drop(['target'],axis=1)
        y = house_data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        X_train, y_train = normalize(X_train, y_train)
        X_test, y_test = normalize(X_test, y_test)
    
        train_data = create_dataset(X_train, y_train)
        test_data = create_dataset(X_test, y_test)
        root_node = build(train_data,split_type='gini')
        train_acc_gini.append(get_accuracy(train_data))
        test_acc_gini.append(get_accuracy(test_data))
        
    plt.hist(train_acc_gini, bins = 30)
    plt.axvline(np.average(train_acc_gini), color='black', linestyle='dashed', linewidth=2)
    plt.title("Accuracy histogram using Gini Index as criterion for Training Data")
    plt.xlabel("Accuracy")
    plt.ylabel("Accuracy Frequency on Training Data")
    plt.show()
    
    std_mean_train_gini = np.std(train_acc_gini) / np.sqrt(np.size(train_acc_gini))
    print("Train accuracy mean : ", np.average(train_acc))
    print("Train accuracy mean standard deviation : ",std_mean_train_gini)
    
    #import seaborn as sns
    plt.hist(test_acc_gini)
    mean_accuracy_gini = np.average(test_acc_gini)
    plt.axvline(x=mean_accuracy,color='blue', ls = '--')
    plt.xlabel("Accuracy")
    plt.ylabel("Accuracy Frequency on Test Data")
    plt.title("Accuracy histogram using Gini Index as criterion for Testing Data")
    plt.show()
    
    std_mean_test_gini = np.std(test_acc_gini) / np.sqrt(np.size(test_acc_gini))
    print("Test accuracy mean : ", mean_accuracy_gini)
    print("Test accuracy mean standard deviation : ",std_mean_test_gini)
