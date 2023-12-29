#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 16:48:15 2022

@author: sahasraiyer
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from math import log2
import sys
import warnings
warnings.filterwarnings("ignore")
import csv
import seaborn as sns
from random_forest_utils import random_forest_algorithm, random_forest_predictions, stratifiedkfold, calculate_accuracy_matrix , createdf, calculate_accuracy_multiclass_dataset

if __name__ == "__main__":
    print("--------------- House Votes Dataset using Entropy -----------------")
    data = pd.read_csv('/Users/sahasraiyer/Downloads/rand_forests/datasets/hw3_house_votes_84.csv',sep=',')
    data = data.rename(columns={'class':'target'})
    df = data.copy()
    feature_cat = [0]*len(data.columns)
    
    df_columns = df.columns
    k = 100
    df_class_combined_k = stratifiedkfold(df, k)
    accuracies = {} 
    precisions = {}
    recalls = {}
    fscores = {}
    max_depth = 5
    n_tree_list = [1,5,10,20,30,40,50]
    
    
    for n_tree in n_tree_list:
        print('Trees in the forest = ',n_tree)
        test = []
        train = []
        each_k_acc = []
        each_k_pre = []
        each_k_rec = []
        each_k_fscore = []
        accuracies[n_tree] = []
        precisions[n_tree] = []
        recalls[n_tree] = []
        fscores[n_tree] = []
        for i in range(k):
            test = df_class_combined_k[i]
            train = []
            for j in range(k):
                if i != j:
                    if len(train) == 0:
                        train = df_class_combined_k[j]
                    else:
                        train = np.vstack((train, df_class_combined_k[j]))
    
            train_full = pd.DataFrame(train, columns=df_columns)
            test_full = pd.DataFrame(test, columns=df_columns)
            
            forest = random_forest_algorithm(train_full, n_trees=n_tree, n_bootstrap=800, n_features=round(len(df_columns)**(1/2)), max_depth=4, split_type='entropy', feature_cat=feature_cat)
            predictions = random_forest_predictions(test_full, forest)
            acc, pre, rec, fscore = calculate_accuracy_matrix(test_full['target'], predictions)

            accuracies[n_tree].append(acc)
            recalls[n_tree].append(rec)
            precisions[n_tree].append(pre)
            fscores[n_tree].append(fscore)

            
            each_k_acc.append(acc)
            each_k_pre.append(pre)
            each_k_rec.append(rec)
            each_k_fscore.append(fscore)
        
        actual_acc = np.asarray(each_k_acc).mean()
        actual_pre = np.asarray(each_k_pre).mean()
        actual_rec = np.asarray(each_k_rec).mean()
        actual_fscore = np.asarray(each_k_fscore).mean()
        
        print('\tAccuracy after {} fold = {}'.format(k,actual_acc))
        print('\tPrecision after {} fold = {}'.format(k,actual_pre))
        print('\tRecall after {} fold = {}'.format(k,actual_rec))
        print('\tF-Score after {} fold = {}'.format(k,actual_fscore))
        
    house_votes_accuracy_df = createdf(accuracies)
    house_votes_accuracy_df.rename(columns={'averages': 'accuracies'}, inplace=True)
    
    plot = sns.catplot(x='n_trees', y='accuracies', data = house_votes_accuracy_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Accuracy plot for House Votes Dataset")
    plt.show()
    
    house_votes_precision_df = createdf(precisions)
    house_votes_precision_df.rename(columns={'averages': 'precision'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='precision', data = house_votes_precision_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Precision plot for House Votes Dataset")
    plt.show()
    
    house_votes_recall_df = createdf(recalls)
    house_votes_recall_df.rename(columns={'averages': 'recall'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='recall', data = house_votes_recall_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Recall plot for House Votes Dataset")
    plt.show()
    
    house_votes_fscore_df = createdf(fscores)
    house_votes_fscore_df.rename(columns={'averages': 'fscore'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='fscore', data = house_votes_fscore_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Fscore plot for House Votes Dataset")
    plt.show()
    
    print("--------------- House Votes Dataset using Gini Index -----------------")
    
    df_class_combined_k = stratifiedkfold(df, k)
    accuracies = {} 
    precisions = {}
    recalls = {}
    fscores = {}
    max_depth = 5
    n_tree_list = [1,5,10,20,30,40,50]
    
    
    for n_tree in n_tree_list:
        print('Trees in the forest = ',n_tree)
        test = []
        train = []
        each_k_acc = []
        each_k_pre = []
        each_k_rec = []
        each_k_fscore = []
        accuracies[n_tree] = []
        precisions[n_tree] = []
        recalls[n_tree] = []
        fscores[n_tree] = []
        for i in range(k):
            test = df_class_combined_k[i]
            train = []
            for j in range(k):
                if i != j:
                    if len(train) == 0:
                        train = df_class_combined_k[j]
                    else:
                        train = np.vstack((train, df_class_combined_k[j]))
    
            train_full = pd.DataFrame(train, columns=df_columns)
            test_full = pd.DataFrame(test, columns=df_columns)
            
            forest = random_forest_algorithm(train_full, n_trees=n_tree, n_bootstrap=800, n_features=round(len(df_columns)**(1/2)), max_depth=4, split_type='gini', feature_cat=feature_cat)
            predictions = random_forest_predictions(test_full, forest)
            acc, pre, rec, fscore = calculate_accuracy_matrix(test_full['target'], predictions)

            accuracies[n_tree].append(acc)
            recalls[n_tree].append(rec)
            precisions[n_tree].append(pre)
            fscores[n_tree].append(fscore)

            
            each_k_acc.append(acc)
            each_k_pre.append(pre)
            each_k_rec.append(rec)
            each_k_fscore.append(fscore)
        
        actual_acc = np.asarray(each_k_acc).mean()
        actual_pre = np.asarray(each_k_pre).mean()
        actual_rec = np.asarray(each_k_rec).mean()
        actual_fscore = np.asarray(each_k_fscore).mean()
        
        print('\tAccuracy after {} fold = {}'.format(k,actual_acc))
        print('\tPrecision after {} fold = {}'.format(k,actual_pre))
        print('\tRecall after {} fold = {}'.format(k,actual_rec))
        print('\tF-Score after {} fold = {}'.format(k,actual_fscore))
        
    house_votes_accuracy_df = createdf(accuracies)
    house_votes_accuracy_df.rename(columns={'averages': 'accuracies'}, inplace=True)
    
    plot = sns.catplot(x='n_trees', y='accuracies', data = house_votes_accuracy_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Accuracy plot for House Votes Dataset")
    plt.show()
    
    house_votes_precision_df = createdf(precisions)
    house_votes_precision_df.rename(columns={'averages': 'precision'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='precision', data = house_votes_precision_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Precision plot for House Votes Dataset")
    plt.show()
    
    house_votes_recall_df = createdf(recalls)
    house_votes_recall_df.rename(columns={'averages': 'recall'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='recall', data = house_votes_recall_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Recall plot for House Votes Dataset")
    plt.show()
    
    house_votes_fscore_df = createdf(fscores)
    house_votes_fscore_df.rename(columns={'averages': 'fscore'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='fscore', data = house_votes_fscore_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Fscore plot for House Votes Dataset")
    plt.show()

    

    print("--------------- Breast Cancer Dataset using Entropy -----------------")
    
    cancer_data = pd.read_csv('/Users/sahasraiyer/Downloads/rand_forests/datasets/hw3_cancer.csv',sep='\t')    
    cancer_data = cancer_data.rename(columns={'Class':'target'})
    s = cancer_data.pop('target')
    cancer_data = pd.concat([cancer_data, s], 1)
    feature_cat = [1]*len(cancer_data.columns)
    
    df = cancer_data.copy()
    df_columns = df.columns
    k = 100
    df_class_combined_k = stratifiedkfold(df, k)
    accuracies = {} 
    precisions = {}
    recalls = {}
    fscores = {}
    max_depth = 5
    n_tree_list = [1,5,10,20,30,40,50]
    
    
    for n_tree in n_tree_list:
        print('Trees in the forest = ',n_tree)
        test = []
        train = []
        each_k_acc = []
        each_k_pre = []
        each_k_rec = []
        each_k_fscore = []
        accuracies[n_tree] = []
        precisions[n_tree] = []
        recalls[n_tree] = []
        fscores[n_tree] = []
        for i in range(k):
            test = df_class_combined_k[i]
            train = []
            for j in range(k):
                if i != j:
                    if len(train) == 0:
                        train = df_class_combined_k[j]
                    else:
                        train = np.vstack((train, df_class_combined_k[j]))
    
            train_full = pd.DataFrame(train, columns=df_columns)
            test_full = pd.DataFrame(test, columns=df_columns)
            
            forest = random_forest_algorithm(train_full, n_trees=n_tree, n_bootstrap=800, n_features=round(len(df_columns)**(1/2)), max_depth=4, split_type='entropy', feature_cat=feature_cat)
            
            predictions = random_forest_predictions(test_full, forest)
            acc, pre, rec, fscore = calculate_accuracy_matrix(test_full['target'], predictions)
            accuracies[n_tree].append(acc)
            recalls[n_tree].append(rec)
            precisions[n_tree].append(pre)
            fscores[n_tree].append(fscore)
            
            each_k_acc.append(acc)
            each_k_pre.append(pre)
            each_k_rec.append(rec)
            each_k_fscore.append(fscore)
        
        actual_acc = np.asarray(each_k_acc).mean()
        actual_pre = np.asarray(each_k_pre).mean()
        actual_rec = np.asarray(each_k_rec).mean()
        actual_fscore = np.asarray(each_k_fscore).mean()

        print('\tAccuracy after {} fold = {}'.format(k,actual_acc))
        print('\tPrecision after {} fold = {}'.format(k,actual_pre))
        print('\tRecall after {} fold = {}'.format(k,actual_rec))
        print('\tF-Score after {} fold = {}'.format(k,actual_fscore))
        
    breast_cancer_accuracy_df = createdf(accuracies)
    breast_cancer_accuracy_df.rename(columns={'averages': 'accuracies'}, inplace=True)
    
    plot = sns.catplot(x='n_trees', y='accuracies', data = breast_cancer_accuracy_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Accuracy plot for Breast Cancer Dataset")
    plt.show()
    
    breast_cancer_precision_df = createdf(precisions)
    breast_cancer_precision_df.rename(columns={'averages': 'precision'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='precision', data = breast_cancer_precision_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Precision plot for Breast Cancer Dataset")
    plt.show()
    
    breast_cancer_recall_df = createdf(recalls)
    breast_cancer_recall_df.rename(columns={'averages': 'recall'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='recall', data = breast_cancer_recall_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Recall plot for Breast Cancer Dataset")
    plt.show()
    
    breast_cancer_fscore_df = createdf(fscores)
    breast_cancer_fscore_df.rename(columns={'averages': 'fscore'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='fscore', data = breast_cancer_fscore_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Fscore plot for Breast Cancer Dataset")
    plt.show()

    
    print("--------------- Wine Dataset using Entropy -----------------")
    wine_data = pd.read_csv('/Users/sahasraiyer/Downloads/rand_forests/datasets/hw3_wine.csv',sep='\t')
    wine_data = wine_data.rename(columns={'# class':'target'})
    s = wine_data.pop('target')
    wine_data = pd.concat([wine_data, s], 1)
    wine_quality = wine_data.target.value_counts(normalize=True)
    wine_quality = wine_quality.sort_index()
    feature_cat = [1]*len(wine_data.columns)
    df = wine_data.copy()

    df_columns = df.columns
    k = 100
    df_class_combined_k = stratifiedkfold(df, k)
    accuracies = {} 
    precisions = {}
    recalls = {}
    fscores = {}
    max_depth = 4
    n_tree_list = [1,5,10,20,30,40,50]

    
    for n_tree in n_tree_list:
        print('Trees in the forest = ',n_tree)
        test = []
        train = []
        accuracies[n_tree] = []
        precisions[n_tree] = []
        recalls[n_tree] = []
        fscores[n_tree] = []
        each_k_acc = []
        each_k_pre = []
        each_k_rec = []
        each_k_fscore = []
        for i in range(k):
            test = df_class_combined_k[i]
            train = []
            for j in range(k):
                if i != j:
                    if len(train) == 0:
                        train = df_class_combined_k[j]
                    else:
                        train = np.vstack((train, df_class_combined_k[j]))
    
            train_full = pd.DataFrame(train, columns=df_columns)
            test_full = pd.DataFrame(test, columns=df_columns)

            if test_full.shape[0] == 0:
                continue
            forest = random_forest_algorithm(train_full, n_trees=n_tree, n_bootstrap=800, n_features=round(len(df_columns)**(1/2)), max_depth=4, split_type='entropy', feature_cat=feature_cat)
            
            predictions = random_forest_predictions(test_full, forest)
            acc, pre, rec, fscore = calculate_accuracy_multiclass_dataset(predictions, test_full.target)
            accuracies[n_tree].append(acc)
            recalls[n_tree].append(rec)
            precisions[n_tree].append(pre)
            fscores[n_tree].append(fscore)
            
            each_k_acc.append(acc)
            each_k_pre.append(pre)
            each_k_rec.append(rec)
            each_k_fscore.append(fscore)
        
        actual_acc = np.asarray(each_k_acc).mean()
        actual_pre = np.asarray(each_k_pre).mean()
        actual_rec = np.asarray(each_k_rec).mean()
        actual_fscore = np.asarray(each_k_fscore).mean()
    
        print('\tAccuracy after {} fold = {}'.format(k,actual_acc)) 
        print('\tPrecision after {} fold = {}'.format(k,actual_pre)) 
        print('\tRecall after {} fold = {}'.format(k,actual_rec)) 
        print('\tFscore after {} fold = {}'.format(k,actual_fscore)) 
    
    wine_dataset_accuracy_df = createdf(accuracies)
    wine_dataset_accuracy_df.rename(columns={'averages': 'accuracies'}, inplace=True)
    
    plot = sns.catplot(x='n_trees', y='accuracies', data = wine_dataset_accuracy_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Accuracy plot for Wine Dataset")
    plt.show()
    
    wine_dataset_precision_df = createdf(precisions)
    wine_dataset_precision_df.rename(columns={'averages': 'precision'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='precision', data =wine_dataset_precision_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Precision plot for Wine Dataset")
    plt.show()
    
    wine_dataset_recall_df = createdf(recalls)
    wine_dataset_recall_df.rename(columns={'averages': 'recall'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='recall', data = wine_dataset_recall_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Recall plot for Wine Dataset")
    plt.show()
    
    wine_dataset_fscore_df = createdf(fscores)
    wine_dataset_fscore_df.rename(columns={'averages': 'fscore'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='fscore', data = wine_dataset_fscore_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Fscore plot for Wine Dataset")
    plt.show()
    
    print("--------------- Wine Dataset using Gini Index -----------------")
    wine_data = pd.read_csv('/Users/sahasraiyer/Downloads/rand_forests/datasets/hw3_wine.csv',sep='\t')
    wine_data = wine_data.rename(columns={'# class':'target'})
    s = wine_data.pop('target')
    wine_data = pd.concat([wine_data, s], 1)
    wine_quality = wine_data.target.value_counts(normalize=True)
    wine_quality = wine_quality.sort_index()
    feature_cat = [1]*len(wine_data.columns)
    df = wine_data.copy()

    df_columns = df.columns
    k = 100
    df_class_combined_k = stratifiedkfold(df, k)
    accuracies = {} 
    precisions = {}
    recalls = {}
    fscores = {}
    max_depth = 4
    n_tree_list = [1,5,10,20,30,40,50]

    
    for n_tree in n_tree_list:
        print('Trees in the forest = ',n_tree)
        test = []
        train = []
        accuracies[n_tree] = []
        precisions[n_tree] = []
        recalls[n_tree] = []
        fscores[n_tree] = []
        each_k_acc = []
        each_k_pre = []
        each_k_rec = []
        each_k_fscore = []
        for i in range(k):
            test = df_class_combined_k[i]
            train = []
            for j in range(k):
                if i != j:
                    if len(train) == 0:
                        train = df_class_combined_k[j]
                    else:
                        train = np.vstack((train, df_class_combined_k[j]))
    
            train_full = pd.DataFrame(train, columns=df_columns)
            test_full = pd.DataFrame(test, columns=df_columns)

            if test_full.shape[0] == 0:
                continue
            forest = random_forest_algorithm(train_full, n_trees=n_tree, n_bootstrap=800, n_features=round(len(df_columns)**(1/2)), max_depth=4, split_type='gini', feature_cat=feature_cat)
            
            predictions = random_forest_predictions(test_full, forest)
            acc, pre, rec, fscore = calculate_accuracy_multiclass_dataset(predictions, test_full.target)
            accuracies[n_tree].append(acc)
            recalls[n_tree].append(rec)
            precisions[n_tree].append(pre)
            fscores[n_tree].append(fscore)
            
            each_k_acc.append(acc)
            each_k_pre.append(pre)
            each_k_rec.append(rec)
            each_k_fscore.append(fscore)
        
        actual_acc = np.asarray(each_k_acc).mean()
        actual_pre = np.asarray(each_k_pre).mean()
        actual_rec = np.asarray(each_k_rec).mean()
        actual_fscore = np.asarray(each_k_fscore).mean()
    
        print('\tAccuracy after {} fold = {}'.format(k,actual_acc)) 
        print('\tPrecision after {} fold = {}'.format(k,actual_pre)) 
        print('\tRecall after {} fold = {}'.format(k,actual_rec)) 
        print('\tFscore after {} fold = {}'.format(k,actual_fscore)) 
    
    wine_dataset_accuracy_df = createdf(accuracies)
    wine_dataset_accuracy_df.rename(columns={'averages': 'accuracies'}, inplace=True)
    
    plot = sns.catplot(x='n_trees', y='accuracies', data = wine_dataset_accuracy_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Accuracy plot for Wine Dataset")
    plt.show()
    
    wine_dataset_precision_df = createdf(precisions)
    wine_dataset_precision_df.rename(columns={'averages': 'precision'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='precision', data =wine_dataset_precision_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Precision plot for Wine Dataset")
    plt.show()
    
    wine_dataset_recall_df = createdf(recalls)
    wine_dataset_recall_df.rename(columns={'averages': 'recall'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='recall', data = wine_dataset_recall_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Recall plot for Wine Dataset")
    plt.show()
    
    wine_dataset_fscore_df = createdf(fscores)
    wine_dataset_fscore_df.rename(columns={'averages': 'fscore'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='fscore', data = wine_dataset_fscore_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Fscore plot for Wine Dataset")
    plt.show()
    
    print("--------------- CMC Dataset using Entropy -----------------")
    
    file = open("/Users/sahasraiyer/Downloads/rand_forests/datasets/cmc.data", encoding='utf-8-sig')
    reader = csv.reader(file, delimiter=',')
    ds = []
    for row in reader:
        row = list(map(int, row))
        ds.append(row)
    file.close()
    columns=['wife_age','wife_education','husband_education','n_children','wife_religion','wife_working','husband_work','SOL_index','media_exposure','contraceptive_method']
    cmc_data = pd.DataFrame(ds,columns=columns)
    
    cmc_data = cmc_data.rename(columns={'contraceptive_method':'target'})
    s = cmc_data.pop('target')
    cmc_data = pd.concat([cmc_data, s], 1)
    
    feature_cat = [1,0,0,1,0,0,0,0,0,0]
    
    df = cmc_data.copy()
    
    df_columns = df.columns
    k = 100
    df_class_combined_k = stratifiedkfold(df, k)
    accuracies = {} 
    precisions = {}
    recalls = {}
    fscores = {}
    max_depth = 4
    n_tree_list = [1,5,10,20,30,40,50]

    split_type = 'entropy'
    
    for n_tree in n_tree_list:
        print('Trees in the forest = ',n_tree)
        test = []
        train = []
        accuracies[n_tree] = []
        precisions[n_tree] = []
        recalls[n_tree] = []
        fscores[n_tree] = []
        each_k_acc = []
        each_k_pre = []
        each_k_rec = []
        each_k_fscore = []
        for i in range(k):
            test = df_class_combined_k[i]
            train = []
            for j in range(k):
                if i != j:
                    if len(train) == 0:
                        train = df_class_combined_k[j]
                    else:
                        train = np.vstack((train, df_class_combined_k[j]))
    
            train_full = pd.DataFrame(train, columns=df_columns)
            test_full = pd.DataFrame(test, columns=df_columns)

            if test_full.shape[0] == 0:
                continue
            forest = random_forest_algorithm(train_full, n_trees=n_tree, n_bootstrap=800, n_features=round(len(df_columns)**(1/2)), max_depth=10, split_type=split_type, feature_cat=feature_cat)
            
            predictions = random_forest_predictions(test_full, forest)
            acc, pre, rec, fscore = calculate_accuracy_multiclass_dataset(predictions, test_full.target)
            accuracies[n_tree].append(acc)
            recalls[n_tree].append(rec)
            precisions[n_tree].append(pre)
            fscores[n_tree].append(fscore)
            
            each_k_acc.append(acc)
            each_k_pre.append(pre)
            each_k_rec.append(rec)
            each_k_fscore.append(fscore)
        
        actual_acc = np.asarray(each_k_acc).mean()
        actual_pre = np.asarray(each_k_pre).mean()
        actual_rec = np.asarray(each_k_rec).mean()
        actual_fscore = np.asarray(each_k_fscore).mean()
    
        print('\tAccuracy after {} fold = {}'.format(k,actual_acc)) 
        print('\tPrecision after {} fold = {}'.format(k,actual_pre)) 
        print('\tRecall after {} fold = {}'.format(k,actual_rec)) 
        print('\tFscore after {} fold = {}'.format(k,actual_fscore)) 
    
    cmc_dataset_accuracy_df = createdf(accuracies)
    cmc_dataset_accuracy_df.rename(columns={'averages': 'accuracies'}, inplace=True)
    
    plot = sns.catplot(x='n_trees', y='accuracies', data = cmc_dataset_accuracy_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Accuracy plot for CMC Dataset")
    plt.show()
    
    cmc_dataset_precision_df = createdf(precisions)
    cmc_dataset_precision_df.rename(columns={'averages': 'precision'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='precision', data =cmc_dataset_precision_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Precision plot for CMC Dataset")
    plt.show()
    
    cmc_dataset_recall_df = createdf(recalls)
    cmc_dataset_recall_df.rename(columns={'averages': 'recall'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='recall', data = cmc_dataset_recall_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Recall plot for CMC Dataset")
    plt.show()
    
    cmc_dataset_fscore_df = createdf(fscores)
    cmc_dataset_fscore_df.rename(columns={'averages': 'fscore'}, inplace=True)
    

    plot = sns.catplot(x='n_trees', y='fscore', data = cmc_dataset_fscore_df, cd='sd', kind = 'point', aspect = 2)
    plot.set(title = "Fscore plot for CMC Dataset")
    plt.show()
    
