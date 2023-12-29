#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 16:53:28 2022

@author: sahasraiyer
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from math import log2
import sys
import warnings
import math
warnings.filterwarnings("ignore")
import csv

#Creating dataframe for ease in plotting

def createdf(accuracy_dict):
    data = []
    for k, acc in accuracy_dict.items():
        if(isinstance(acc, list)):
            for v in acc:
                data.append([k, v])

    df = pd.DataFrame(data, columns = ['n_trees', 'averages'])
    return df

def calculate_accuracy_matrix(actual, prediction):
    # both are series
    ac = actual.values.tolist()
    pr = prediction.values.tolist()
    tp, tn, fp, fn = 0,0,0,0
    tot = len(ac)
    for a,b in zip(ac,pr):
        #print(a,b)
        if (a==1 and b==1) :
            tp += 1
        elif (a==0 and b==0):
            tn += 1
        elif (a==1 and b==0):
            fn += 1
        elif (a==0 and b==1):
            fp += 1
    #print(tp, fp)
    accuracy = ((tp+tn)/tot)*100
    precision = (tp / (tp+fp))*100 if tp > 0 else 0
    recall = (tp/(tp+fn))*100 if tp > 0 else 0
    f_score = (2*(precision * recall))/(precision+recall) if precision>0 and recall>0 else 0
    return accuracy, precision, recall, f_score

def calculate_accuracy(actual, prediction):
    # both are series
    ac = actual.values.tolist()
    pr = prediction.values.tolist()
    cor = 0
    tot = len(ac)
    for a,b in zip(ac,pr):
        if a==b:
            cor+=1
    accuracy = (cor/tot)*100
    return accuracy

def random_forest_predictions(test_df, forest):
    df_predictions = {}
    # .values.tolist()
#     print(test_df)
    for i in range(len(forest)):
        col = "tree_{}".format(i)
        root_node = forest[i]
        predictions = decision_tree_predictions(test_df, tree=root_node)
#         print(type(predictions), len(predictions))
        # create a df of all tree predictions
#         print(col,len(predictions),end='')
        if len(predictions) < 1:
#             print('TEST_DF')
#             print(test_df)
            
#             print('COL',col)
#             print('Prediction')
            print(predictions)
        df_predictions[col] = predictions
#     print(df_predictions)
    df_predictions = pd.DataFrame(df_predictions)
#     print(df_predictions.shape)
    
    random_forest_predictions = df_predictions.mode(axis=1)[0]
    
    return random_forest_predictions

def random_forest_algorithm(train_data, n_trees, max_depth, n_bootstrap, n_features, split_type, feature_cat):
#     print(split_type)
    # n_features is root(features)
    forest = []
    for i in range(n_trees):
        df_bootstrap_df = bootstrapping(train_data, n_bootstrap)
        depth = 0
        tree = decision_tree_algorithm(df_bootstrap_df, max_depth=max_depth, random_subspace=n_features, feature_cat=feature_cat, min_samples = 3)
#         tree = build(df_bootstrap_df, max_depth=max_depth, random_subspace=n_features, feature_cat=feature_cat, split_type=split_type)
        forest.append(tree)
    return forest

def bootstrapping(train_df, n):
    resample_ratio = 0.1
    indices = np.random.randint(low=2, high=len(train_df), size=n)
#     print(indices)
    random = np.random.choice(indices, size=round(resample_ratio*n), replace=False)
#     print(random)
    exactindices = np.delete(indices, np.where(indices == random))
#     print(exactindices)
    fill = np.random.choice(exactindices, size=round(resample_ratio*n), replace=False)
#     print(fill)
    final_indices = np.concatenate((exactindices, fill))
#     print(final_indices)
    df_boost = train_df.iloc[final_indices]
    return df_boost

def decision_tree_predictions(test_df, tree):
    preds = test_df.apply(predict_example, args=(tree,), axis=1)
    return preds
#mark
def predict_example(example, tree):
    qs = list(tree.keys())[0]
    feature_name, com_op, val = qs.split(" ")

    # ask question
    if com_op == "<=":
        if example[feature_name] <= float(val):
            ans = tree[qs][0]
        else:
            ans = tree[qs][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == val:
            ans = tree[qs][0]
        else:
            ans = tree[qs][1]

    # base case
    if not isinstance(ans, dict):
        return ans
    
    # recursive part
    else:
        res_tree = ans
        return predict_example(example, res_tree)

#mark
def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5, random_subspace=None,feature_cat=[],split_type='entropy'):
    if counter == 0:
        global COL_HEADERS, FEATURE_TYPE
        COL_HEADERS = df.columns
        FEATURE_TYPE = det_feature(df,feature_cat)
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        return classification

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        unique_splits = get_potential_splits(data, random_subspace,feature_cat)
        split_col, split_val = determine_best_split(data, unique_splits,split_type)
        bel, abv = split_data(data, split_col, split_val)
        
        # check for empty data
        if len(bel) == 0 or len(abv) == 0:
            classification = classify_data(data)
            return classification
        
        # determine question
        feature_name = COL_HEADERS[split_col]
        type_feature = FEATURE_TYPE[split_col]
        if type_feature == "numerical":
            qs = "{} <= {}".format(feature_name, split_val)
            
        # feature is categorical
        else:
            qs = "{} = {}".format(feature_name, split_val)
        
        # instantiate sub-tree
        sub_tree = {qs: []}
        
        # find answers (recursion)
        yes_ans = decision_tree_algorithm(bel, counter, min_samples, max_depth, random_subspace,feature_cat=feature_cat,split_type=split_type)
        no_ans = decision_tree_algorithm(abv, counter, min_samples, max_depth, random_subspace,feature_cat=feature_cat,split_type=split_type)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_ans == no_ans:
            sub_tree = yes_ans
        else:
            sub_tree[qs].append(yes_ans)
            sub_tree[qs].append(no_ans)
        
        return sub_tree

#mark
def get_potential_splits(data, random_subspace, feature_cat):
    
    unique_splits = {}
    _, n_cols = data.shape
    col_idxs = list(range(n_cols - 1))    
    
    if random_subspace and random_subspace <= len(col_idxs):
        col_idxs = random.sample(population=col_idxs, k=random_subspace)
    for col_idx in col_idxs:          
        vals = data[:, col_idx]
#         print(column_index,len(feature_cat))
        if feature_cat[col_idx] == 0:
            unique_vals = np.unique(vals)
        else:
            unique_vals = [np.mean(vals)]
            
        
        unique_splits[col_idx] = unique_vals
    
    return unique_splits

#mark
def split_data(data, split_col, split_val):
    
    split_col_val = data[:, split_col]

    type_feature = FEATURE_TYPE[split_col]
    if type_feature == "numerical":
        bel = data[split_col_val <= split_val]
        abv = data[split_col_val >  split_val]
    
    # feature is categorical   
    else:
        bel = data[split_col_val == split_val]
        abv = data[split_col_val != split_val]
    
    return bel, abv

def query(row, feature, value):
#     print(row)
    com_val = row[feature]
#     print(row[feature], value)
#     print(com_val, type(com_val))
    if com_val >= value:
        return True
    else:
        return False

# Partitions the data into left and right based on who satisfies row[feature]>=value
def partition(data, feature, value):
    true_r, false_r = [],[]
    
#     for index,row in data.iterrows():
    for index,row in enumerate(data):
        if query(row, feature, value):
            true_r.append(row)
        else:
            false_r.append(row)

    return true_r, false_r

def determine_best_split(data, unique_splits,split_type):
    
    entropy = 9999
    for col_idx in unique_splits:
        for val in unique_splits[col_idx]:
            bel, abv = split_data(data, split_col=col_idx, split_val=val)
            curr_entropy = infogain(bel, abv, split_type)
            
            if curr_entropy <= entropy:
                entropy = curr_entropy
                best_split_col = col_idx
                best_split_val = val
    
    return best_split_col, best_split_val

def infogain(data_below, data_above, split_type):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    if split_type == 'entropy':
        overall_entropy =  (p_data_below * entropy(data_below) + p_data_above * entropy(data_above))
    elif split_type=='gini':
        overall_entropy =  (p_data_below * gini(data_below) + p_data_above * gini(data_above))
    
    return overall_entropy

def entropy_check(data):
    count = {}
    
    for row in data:
#         print(row)
        label = row[-1]
        if label not in count:
            count[label] = 0
        count[label]+=1
    total = float(len(data))
    keys = len(list(count.keys()))
    
    # uncomment this
    classes = 2
    if keys != classes:
        return 0
#     if (0 not in count.keys()) and (1 not count.keys()):
#         return 0
    entropy = 0
    for cl, val in count.items():
        p = val/float(len(data))
        entropy += -p*log2(p)
        
#     class0 = count[0]/float(len(data))
#     class1 = count[1]/float(len(data))
#     entropy = -(class0 * log2(class0) + class1 * log2(class1))
#     print('entropy',entropy, entropyx)
    return entropy

def entropy(data):
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy

def gini(data):
    count = {}
    # if it was dataframe
    # count = data['target'].value_counts().to_dict()
    
    # if list
    for row in data:
        label = row[-1]
        if label not in count:
            count[label] = 0
        count[label]+=1
    impurity = 1
#     print(count)
    for label in count:
        prob = count[label]/float(len(data))
        impurity -= prob**2
#     print(impurity)
    return impurity

def classify_data(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification


def check_purity(data):
    
    label_column = data[:, -1]
    unq = np.unique(label_column)

    if len(unq) == 1:
        return True
    else:
        return False

def det_feature(df, feature_cat):
    # passed a feature cat
    feature_types = []
    if len(feature_cat) > 0:
        for a in feature_cat:
            if a==0:
                feature_types.append("categorical")
            else:
                feature_types.append("numerical")

    return feature_types

def calculate_accuracy_multiclass_dataset(actual, prediction):
    # both are series
    ac = pd.Series(actual, name = 'Actual')
    pr = pd.Series(prediction, name = 'Pred')
    cor = 0
    #TP, TN, FP, FN = [],[],[],[]
    tp, tn, fp, fn = [],[],[],[]
    #tp_1, tp_2, tp_3, fp_1, fn_1, fp_2, fn_2, fp_3, fn_3 = 0,0,0,0,0,0,0,0,0
    tot = len(ac)
    tp.append(np.sum(np.logical_and(pr == 1, ac == 1)))
    tp.append(np.sum(np.logical_and(pr == 2, ac == 2)))
    tp.append(np.sum(np.logical_and(pr == 3, ac == 3)))
    tn.append(np.sum(np.logical_and(pr == 3, ac == 3)) + np.sum(np.logical_and(pr == 2, ac == 2)))
    tn.append(np.sum(np.logical_and(pr == 3, ac == 3)) + np.sum(np.logical_and(pr == 1, ac == 1)))
    tn.append(np.sum(np.logical_and(pr == 2, ac == 2)) + np.sum(np.logical_and(pr == 1, ac == 1)))
    fp.append(np.sum(np.logical_and(pr == 1, ac == 2)) + np.sum(np.logical_and(pr == 1, ac == 3)))
    fp.append(np.sum(np.logical_and(pr == 3, ac == 1)) + np.sum(np.logical_and(pr == 3, ac == 2)))
    fp.append(np.sum(np.logical_and(pr == 2, ac == 1)) + np.sum(np.logical_and(pr == 2, ac == 3)))
    fn.append(np.sum(np.logical_and(pr == 2, ac == 1)) + np.sum(np.logical_and(pr == 3, ac == 1)))
    fn.append(np.sum(np.logical_and(pr == 3, ac == 2)) + np.sum(np.logical_and(pr == 1, ac == 2)))
    fn.append(np.sum(np.logical_and(pr == 1, ac == 3)) + np.sum(np.logical_and(pr == 2, ac == 3)))
    accuracy = (np.sum(tp)+ np.sum(tn))/(np.sum(tp)+ np.sum(tn) + np.sum(fp)+ np.sum(fn))
    precision = (np.sum(tp))/(np.sum(tp) + np.sum(fp))

    recall = (np.sum(tp))/(np.sum(tp) + np.sum(fn))

    fscore = (2 * (precision * recall)/(precision + recall)) 
    #fscore = fscore if !math.isnan(fscore) else 1
    if math.isnan(fscore):
        fscore = 1
    return accuracy, precision, recall, fscore


def stratifiedkfold(df, k):
    nclasses = len(df['target'].value_counts().index)
    classes = list(df['target'].value_counts().index)
    classratio = list(df['target'].value_counts(normalize=True).values)
    totaldata = len(df['target'])
#     print(nclasses, classes, classratio, totaldata,k)

    kdf = df.copy()
    
    # ndf = kdf.values.tolist()
    df_partition = {}
    safa = None
    for index, cl in enumerate(classes):
        safa = df[df['target'] == cl]
        ndsafa = safa.values.tolist()
        splitted = np.array_split(ndsafa, k)
        df_partition[cl] = splitted

    df_class_combined_k = {}
    totx = 0
    for i in range(k):
#         print(i,'th partition in every class')
        singleclass = [] 
        for cl in classes:
            totx+=len(df_partition[cl][i])
#             print(len(df_partition[cl][i]))
            if  len(singleclass) == 0:
                singleclass = df_partition[cl][i]
            else:
                singleclass = np.vstack((singleclass, df_partition[cl][i]))
        # shuffule it nice
        np.random.shuffle(singleclass)
        df_class_combined_k[i] = singleclass
    return df_class_combined_k
