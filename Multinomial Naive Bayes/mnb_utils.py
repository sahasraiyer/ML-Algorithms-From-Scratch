import re
import os
import glob
import random
from nltk.corpus import stopwords
import nltk
import math
import numpy as np

REPLACE_NO_SPACE = re.compile("[._;:!`¦\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
nltk.download('stopwords')  

def preprocess_text(text):
	stop_words = set(stopwords.words('english'))
	text = REPLACE_NO_SPACE.sub("", text)
	text = REPLACE_WITH_SPACE.sub(" ", text)
	text = re.sub(r'\d+', '', text)
	text = text.lower()
	words = text.split()
	return [w for w in words if w not in stop_words]

def load_training_set(percentage_positives, percentage_negatives):
	vocab = set()
	positive_instances = []
	negative_instances = []  
	for filename in glob.glob('/Users/sahasraiyer/Downloads/hw2/train/pos/*.txt'):
		if random.random() > percentage_positives:
			continue
		with open(os.path.join(os.getcwd(), filename), 'r') as f:
			contents = f.read()
			contents = preprocess_text(contents)
			positive_instances.append(contents)
			vocab = vocab.union(set(contents))
	for filename in glob.glob('/Users/sahasraiyer/Downloads/hw2/train/neg/*.txt'):
		if random.random() > percentage_negatives:
			continue
		with open(os.path.join(os.getcwd(), filename), 'r') as f:
			contents = f.read()
			contents = preprocess_text(contents)
			negative_instances.append(contents)
			vocab = vocab.union(set(contents))	
	return positive_instances, negative_instances, vocab

def load_test_set(percentage_positives, percentage_negatives):
	positive_instances = []
	negative_instances = []
	for filename in glob.glob('/Users/sahasraiyer/Downloads/hw2/test/pos/*.txt'):
		if random.random() > percentage_positives:
			continue
		with open(os.path.join(os.getcwd(), filename), 'r') as f:
			contents = f.read()
			contents = preprocess_text(contents)
			positive_instances.append(contents)
	for filename in glob.glob('/Users/sahasraiyer/Downloads/hw2/test/neg/*.txt'):
		if random.random() > percentage_negatives:
			continue
		with open(os.path.join(os.getcwd(), filename), 'r') as f:
			contents = f.read()
			contents = preprocess_text(contents)
			negative_instances.append(contents)
	return positive_instances, negative_instances

def create_dictionary(data) : 
    #Creates a distionary of the dataset being passed to it
    review_dict = {}
    for word_vec in data:
        for word in word_vec:
            if word not in review_dict:
                review_dict[word]=1
            else:
                review_dict[word]+=1
    return review_dict


def mnb_using_log(pos_test_set, pos_dict, perc_pos, pos_words, neg_test_set, neg_dict, perc_neg, neg_words):
    tp, tn, fp, fn = 0, 0, 0, 0
    #y_pred = [0] * (len(pos_test_set)+len(neg_test_set))
    #Multinomial Naive Bayes using posterior log probabilities
    for i in range(len(pos_test_set)):
        unique_words = set(pos_test_set[i])
        second_term_pos, second_term_neg = 0, 0
        for word in unique_words:
            if word in pos_dict:
                second_term_pos += math.log(pos_dict[word] / pos_words)
            if word in neg_dict:
                second_term_neg += math.log(neg_dict[word] / neg_words)
        log_doc_pos = math.log(perc_pos) + second_term_pos
        log_doc_neg = math.log(perc_neg) + second_term_neg
        if log_doc_pos>=log_doc_neg:
            #y_pred[i] = 1
            #correct_count+=1
            tp+=1
        else:
            fn+=1

    i +=1
    for word_vec in neg_test_set:
        unique_words = set(word_vec)
        second_term_pos, second_term_neg = 0, 0
        for word in unique_words:
            if word in pos_dict:
                second_term_pos += math.log(pos_dict[word] / pos_words)
            if word in neg_dict:
                second_term_neg += math.log(neg_dict[word] / neg_words)
        log_doc_pos = math.log(perc_pos) + second_term_pos
        log_doc_neg = math.log(perc_neg) + second_term_neg
        if log_doc_pos<log_doc_neg:
            #correct_count+=1
            tn+=1
        else:
            #y_pred[i] = 1
            fp +=1
        i+=1
    return tp, tn, fp, fn


def mnb_using_posterior(pos_test_set, pos_dict, perc_pos, pos_words, neg_test_set, neg_dict, perc_neg, neg_words):
    tp, tn, fp, fn = 0, 0, 0, 0
    y_pred = [0] * (len(pos_test_set)+len(neg_test_set))
    #Multinomial Naive Bayes using posterior probability formula
    for i in range(len(pos_test_set)):
        unique_words = set(pos_test_set[i])
        second_term_pos, second_term_neg = 1.0, 1.0
        for word in unique_words:
            if word in pos_dict:
                second_term_pos *= pos_dict[word] / pos_words
            if word in neg_dict:
                second_term_neg *= neg_dict[word] / neg_words
        doc_pos = perc_pos * second_term_pos
        doc_neg = perc_neg * second_term_neg
        if doc_pos>=doc_neg:
            #y_pred[i] = 1
            #correct_count+=1
            tp+=1
        else:
            fn+=1

    i +=1
    for word_vec in neg_test_set:
        unique_words = set(word_vec)
        second_term_pos, second_term_neg = 1.0, 1.0
        for word in unique_words:
            if word in pos_dict:
                second_term_pos *= pos_dict[word] / pos_words
            if word in neg_dict:
                second_term_neg *= neg_dict[word] / neg_words
        doc_pos = perc_pos * second_term_pos
        doc_neg = perc_neg * second_term_neg
        if doc_pos<doc_neg:
            #correct_count+=1
            tn+=1
        else:
            #y_pred[i] = 1
            fp +=1
        #i+=1
    return tp, tn, fp, fn


def mnb_using_alpha(pos_test_set, pos_dict, perc_pos, pos_words, neg_test_set, neg_dict, perc_neg, neg_words, vocab, alpha = 1):
    tp, tn, fp, fn = 0, 0, 0, 0
    #y_pred = [0] * (len(pos_test_set)+len(neg_test_set))
    #Multinomial Naive Bayes using Laplace Smoothing
    for word_vec in pos_test_set:
        unique_words = set(word_vec)
        second_term_pos, second_term_neg = 0, 0
        for word in unique_words:
            pos_val = pos_dict[word] if pos_dict.get(word) != None else 0
            neg_val = neg_dict[word] if neg_dict.get(word) != None else 0
            second_term_pos += math.log((pos_val + alpha) / (pos_words + vocab * alpha))
            second_term_neg += math.log((neg_val + alpha) / (neg_words + vocab * alpha))
        log_doc_pos = math.log(perc_pos) + second_term_pos
        log_doc_neg = math.log(perc_neg) + second_term_neg
        if log_doc_pos>=log_doc_neg:
            #y_pred[i] = 1
            #correct_count+=1
            tp+=1
        else:
            fn+=1

    #i +=1
    for word_vec in neg_test_set:
        unique_words = set(word_vec)
        second_term_pos, second_term_neg = 0, 0
        for word in unique_words:
            pos_val = pos_dict[word] if pos_dict.get(word) != None else 0
            neg_val = neg_dict[word] if neg_dict.get(word) != None else 0
            second_term_pos += math.log((pos_val + alpha) / (pos_words + vocab * alpha))
            second_term_neg += math.log((neg_val + alpha) / (neg_words + vocab * alpha))
        log_doc_pos = math.log(perc_pos) + second_term_pos
        log_doc_neg = math.log(perc_neg) + second_term_neg
        if log_doc_pos<log_doc_neg:
            #correct_count+=1
            tn+=1
        else:
            #y_pred[i] = 1
            fp +=1
        #i+=1
    return tp, tn, fp, fn


def print_metric_values(tp, tn, fp, fn):
    print("Precision : ", tp/(tp+fp))
    print("Recall : ", tp/(tp+fn))
    print("Accuracy : ", (tp+tn)/(tp+tn+fp+fn))
    print("Confusion matrix : ")
    print(np.array([[tp, fn], [fp, tn]]))
		
