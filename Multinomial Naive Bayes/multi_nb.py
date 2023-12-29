#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 23:34:38 2022

@author: sahasraiyer
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from mnb_utils import load_training_set, load_test_set, create_dictionary, mnb_using_posterior, mnb_using_log, mnb_using_alpha, print_metric_values


if __name__ == "__main__":
    train_pos, train_neg, train_vocab = load_training_set(0.2, 0.2)
    test_pos, test_neg = load_test_set(0.2, 0.2)
    
    train_pos_dict = create_dictionary(train_pos)
    train_neg_dict = create_dictionary(train_neg)
    
    perc_pos_docs = len(train_pos)/(len(train_pos) + len(train_neg))
    perc_neg_docs = len(train_neg)/(len(train_pos) + len(train_neg))
    
    pos_words = sum(train_pos_dict.values())
    neg_words = sum(train_neg_dict.values())
    
    #Classification and accuracies using posterior probabilities
    tp, tn, fp, fn = mnb_using_posterior(test_pos, train_pos_dict, perc_pos_docs, pos_words, test_neg, train_neg_dict, perc_neg_docs, neg_words)
    print("Accuracies using posterior probability : ")
    print_metric_values(tp, tn, fp, fn)
    
    #Classification and accuracies using log-posterior probabilities
    
    tp, tn, fp, fn = mnb_using_log(test_pos, train_pos_dict, perc_pos_docs, pos_words, test_neg, train_neg_dict, perc_neg_docs, neg_words)
    print("Accuracies using log posterior probability : ")
    print_metric_values(tp, tn, fp, fn)

    
    #Load training and test set again for computing using laplace smoothing alpha = 1

    train_pos, train_neg, train_vocab = load_training_set(0.2, 0.2)
    test_pos, test_neg = load_test_set(0.2, 0.2)
    
    train_pos_dict = create_dictionary(train_pos)
    train_neg_dict = create_dictionary(train_neg)
    
    perc_pos_docs = len(train_pos)/(len(train_pos) + len(train_neg))
    perc_neg_docs = len(train_neg)/(len(train_pos) + len(train_neg))
    
    pos_words = sum(train_pos_dict.values()) #len(train_pos_dict) 
    neg_words = sum(train_neg_dict.values())
    

    tp, tn, fp, fn = mnb_using_alpha(test_pos, train_pos_dict, perc_pos_docs, pos_words, test_neg, train_neg_dict, perc_neg_docs, neg_words, len(train_vocab),1)
    print("Accuracies using log posterior probability and alpha = 1 : ")
    print_metric_values(tp, tn, fp, fn)
    
    
    #Laplace smoothing using a range of alpha values
    alpha = 0.0001
    alpha_vals = []
    while alpha<1001:
        alpha_vals.append(alpha)
        alpha *= 10
    
    alpha_vals
    
    alpha_accuracy = {}

    for alpha in alpha_vals:
        tp, tn, fp, fn = mnb_using_alpha(test_pos, train_pos_dict, perc_pos_docs, pos_words, test_neg, train_neg_dict, perc_neg_docs, neg_words, len(train_vocab),alpha)
        alpha_accuracy[alpha] = (tp+tn)/(tp+tn+fp+fn)
        
    #print(alpha_accuracy)
    
    plt.plot(alpha_accuracy.keys(), alpha_accuracy.values())
    plt.xscale("log")
    plt.show()
    
    
    #Using best performing alpha and running with entire training and test set 
    best_performing_aplha = max(alpha_accuracy, key = alpha_accuracy.get)
    print("Best performing alpha is : ",best_performing_aplha)
    
    
    
    train_pos, train_neg, train_vocab = load_training_set(1.0, 1.0)
    test_pos, test_neg = load_test_set(1.0, 1.0)
    
    train_pos_dict = create_dictionary(train_pos)
    train_neg_dict = create_dictionary(train_neg)
    
    perc_pos_docs = len(train_pos)/(len(train_pos) + len(train_neg))
    perc_neg_docs = len(train_neg)/(len(train_pos) + len(train_neg))
    
    pos_words = sum(train_pos_dict.values()) #len(train_pos_dict) 
    neg_words = sum(train_neg_dict.values())
    
    tp, tn, fp, fn = mnb_using_alpha(test_pos, train_pos_dict, perc_pos_docs, pos_words, test_neg, train_neg_dict, perc_neg_docs, neg_words, len(train_vocab), best_performing_aplha)
    print("Accuracies using log posterior probability and best performing alpha value {}: ".format(best_performing_aplha))
    print_metric_values(tp, tn, fp, fn)
    
    #Using only 50% of the training dataset
    train_pos, train_neg, train_vocab = load_training_set(0.5, 0.5)
    test_pos, test_neg = load_test_set(1.0, 1.0)
    
    train_pos_dict = create_dictionary(train_pos)
    train_neg_dict = create_dictionary(train_neg)
    
    perc_pos_docs = len(train_pos)/(len(train_pos) + len(train_neg))
    perc_neg_docs = len(train_neg)/(len(train_pos) + len(train_neg))
    
    pos_words = sum(train_pos_dict.values()) #len(train_pos_dict) 
    neg_words = sum(train_neg_dict.values())
    
    tp, tn, fp, fn = mnb_using_alpha(test_pos, train_pos_dict, perc_pos_docs, pos_words, test_neg, train_neg_dict, perc_neg_docs, neg_words, len(train_vocab),best_performing_aplha)
    print("Accuracies using log posterior probability and best performing alpha value {}, using 50% dataset and 100% test dataset: ".format(best_performing_aplha))
    print_metric_values(tp, tn, fp, fn)

    
    #Unbalanced training dataset
    #best_performing_aplha = 10
    train_pos, train_neg, train_vocab = load_training_set(0.1, 0.5)
    test_pos, test_neg = load_test_set(1.0, 1.0)
    
    train_pos_dict = create_dictionary(train_pos)
    train_neg_dict = create_dictionary(train_neg)
    
    perc_pos_docs = len(train_pos)/(len(train_pos) + len(train_neg))
    perc_neg_docs = len(train_neg)/(len(train_pos) + len(train_neg))
    
    pos_words = sum(train_pos_dict.values()) #len(train_pos_dict) 
    neg_words = sum(train_neg_dict.values())
    
    tp, tn, fp, fn = mnb_using_alpha(test_pos, train_pos_dict, perc_pos_docs, pos_words, test_neg, train_neg_dict, perc_neg_docs, neg_words, len(train_vocab),best_performing_aplha)
    print("Accuracies using unbalanced dataset (10% positive instances, 50% negative instances): ")
    print_metric_values(tp, tn, fp, fn)
    
