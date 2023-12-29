#!/usr/bin/env python
# coding: utf-8

# In[164]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score, f1_score


# In[165]:


def initialize_network(n_neurons):
    network = {}
    for n in range(len(n_neurons)-1):
        network['W'+str(n+1)] = np.random.rand(n_neurons[n+1],n_neurons[n]+1)
    num_layers = len(n_neurons)
    return network, num_layers


# In[166]:


def sigmoid(z):
    sigmoid = 1.0/(1.0+np.exp(-z))
    return sigmoid


# In[167]:


# Forward pass
def forward_propagation(network, num_layers, data, n_features):
    layer_computations = []

    for i, row in enumerate(data):
        layer_comp = {}
        layer_comp['A0'] = row[:n_features]
        layer_comp['A0'] = np.insert(layer_comp['A0'], 0, 1)

        for i in range(num_layers-1):

            layer_comp['Z'+str(i+1)] = np.dot(network['W'+str(i+1)],layer_comp['A'+str(i)])
            # activation
            layer_comp['A'+str(i+1)] = sigmoid(layer_comp['Z'+str(i+1)])
            if i!=num_layers-2:
                layer_comp['A'+str(i+1)] = np.insert(layer_comp['A'+str(i+1)],0,1)
                
        layer_computations.append(layer_comp)
    return layer_computations


# In[168]:


def back_propagation(network, layer_computations, data, n_features):
    deltas = []
    gradients = []
    p_vals = []
    gradients_byweights = {}
    p = {}
    for row, layer_comp in zip(data, layer_computations):
        layer_count = len(network)
        delta = {}
        grads = {}
        els = layer_comp.items()
        #final_output_key = str(list(comps[-1].keys())[-1])
        fx = list(els)[-1][1]
        y = np.array(row[n_features:])
        delta['delta'+str(layer_count)] = np.array(fx-y)
        for i in range(len(network)-1, 0, -1):
            delta['delta'+str(i)] = np.multiply(np.multiply(np.dot(np.array(network['W'+str(i+1)]).T,delta['delta'+str(i+1)]),np.array(layer_comp['A'+str(i)])[:, np.newaxis].T),np.array(1-np.array(layer_comp['A'+str(i)])[:, np.newaxis].T))[0][1:]  
        deltas.append(delta)
        for i in range(len(network), 0, -1):
            grads['grad'+str(i)] = 0
            shape_ = layer_comp['A'+str(i-1)][:, np.newaxis].T.shape
            if len(delta['delta'+str(i)].shape) == len(shape_):
                grads['grad'+str(i)] = grads['grad'+str(i)] + np.dot(delta['delta'+str(i)], layer_comp['A'+str(i-1)][:, np.newaxis].T)
            else:
                grads['grad'+str(i)] = grads['grad'+str(i)] + np.dot(np.array([delta['delta'+str(i)]]).T, layer_comp['A'+str(i-1)][:, np.newaxis].T)
        gradients.append(grads)
 
    data_len = len(data)
    len_network = len(network)

    counter = collections.Counter()
    for grad in gradients: 
        counter.update(grad)
    result = dict(counter)
            
        
    for i in range(len(network), 0, -1): 
        p['p'+str(i)] = np.multiply(lambd, network['W'+str(i)])
        p['p'+str(i)][:,0] = 0
        gradients_byweights['grad'+str(i)] = (1/data_len)*(result['grad'+str(i)] + p['p'+str(i)])
    return gradients_byweights
       
#grad_bywt = backprop(network, comps, data, n_features=2)


# In[169]:


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
    #print(accuracy, precision, recall, fscore)
    #fscore = fscore if !math.isnan(fscore) else 1
    if math.isnan(fscore):
        fscore = 1
    return accuracy, precision, recall, fscore


# In[170]:


def calculate_accuracy(actual, prediction):
    # both are series
    ac = actual
    pr = prediction
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
    #print(tp, fp, tn, fn)
    accuracy = ((tp+tn)/tot)*100
    precision = (tp / (tp+fp))*100 if tp>0 else 100
    recall = (tp/(tp+fn))*100
    f_score = (2*(precision * recall))/(precision+recall)
    return accuracy, precision, recall, f_score


# In[171]:


def cost_function(data, layer_computations, network, n_features, lambd):
    cost = 0
    #print("Lambda : ",lambd)
    for row, layer_comp in zip(data, layer_computations):
        els = layer_comp.items()
        #final_output_key = str(list(els[-1].keys())[-1])

        fx = list(els)[-1][1]

        y = np.array(row[n_features:])
#         print("y : ",y)
#         print("fx : ",fx)

        #cost_curr = sum(-(y*np.log(fx)) - (1-np.array(y))*np.log(1-np.array(fx)))
        cost_curr = - np.multiply(y,np.log(fx)) - np.multiply(1-y,np.log(1-np.array(fx)))
        cost += np.sum(cost_curr)
        if cost < 0:
            print(layer_comp, row)
            print("fx : {} \ty : {} \tcost : {} \tcost_curr: {}".format(fx, y, cost, np.sum(cost_curr)))
            assert 0
    #print(cost)
    #print("Length of data : ",len(data))
    cost = cost/len(data)
    S=0
    for key, values in network.items():
        #if(isinstance(values, list)):
            for value in values:
                #if(isinstance(value, list)):
                    for val in value:
                        S+=np.power(val, 2)
    S = (lambd/(2*(len(data))))*S
    return (cost+S)


# In[172]:


def train_network(network, data,  num_layers, n_features, epochs, lambd, alpha):
    cost_dict = {}
    for epoch in range(epochs):
        # feed forward
        layer_comps = forward_propagation(network, num_layers, data, n_features)
        # cost function - print cost
        cost = cost_function(data, layer_comps, network, n_features, lambd)
        print("Cost in epoch {} is {}".format(epoch, cost))
        cost_dict[epoch] = cost
        # backpropagation
        grad_bywt = back_propagation(network, layer_comps, data, n_features)
        # weight updates
        #print(grad_bywt)
        for i in range(1, len(network)+1):
            network['W'+str(i)] = network['W'+str(i)] - alpha * grad_bywt['grad'+str(i)]
    return cost_dict


# In[173]:


def normalize(features):
    #labels = labels.to_numpy()
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    return features_norm


# In[174]:


def stratifiedkfold(df, k):
    nclasses = len(df['Class'].value_counts().index)
    classes = list(df['Class'].value_counts().index)
    classratio = list(df['Class'].value_counts(normalize=True).values)
    totaldata = len(df['Class'])
#     print(nclasses, classes, classratio, totaldata,k)

    kdf = df.copy()
    
    # ndf = kdf.values.tolist()
    df_partition = {}
    safa = None
    for index, cl in enumerate(classes):
        safa = df[df['Class'] == cl]
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


# In[175]:


def normalize_df(train_full, test_full):
    train_full.iloc[:,:n_features] = train_full.iloc[:,:n_features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    test_full.iloc[:,:n_features] = test_full.iloc[:,:n_features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return train_full.to_numpy(), test_full.to_numpy()


# In[177]:


data = pd.read_csv('/Users/sahasraiyer/Downloads/hw4/datasets/hw3_cancer.csv')


# In[178]:


k=10
df_class_combined_k = stratifiedkfold(data, k)
df_columns = data.columns


# In[179]:


lambd = 0.01
accuracy_byfold = []
f1score_byfold = []
accuracies = []
n_features=9
for i in range(10):
    correct, total = 0, 0
    accuracy = {}
    test = df_class_combined_k[i%10]
    train = []
    preds = []
    actuals = []
    #print(len(df_class_combined_k))
    for j in range(10):
        if i != j:
            if len(train) == 0:
                train = df_class_combined_k[j]
            else:
                train = np.vstack((train, df_class_combined_k[j]))
    #print(train.shape, df_columns.shape)
    #print("Length of train data in main loop : ",len(train_full))
    train_full = pd.DataFrame(train, columns=df_columns)
    #train_full = shuffle(train_full)
    train_full = pd.get_dummies(train_full, columns=['Class'])
    #print(train_full)
    test_full = pd.DataFrame(test, columns=df_columns)
    #test_full = shuffle(test_full)
    test_full = pd.get_dummies(test_full, columns=['Class'])
    
    #train_full = normalize(train_full)
    #print(train_full)
    #test_full = normalize(test_full)
    train_full, test_full = normalize_df(train_full, test_full)
    network, num_layers = initialize_network([9,15,20,2]) #[13,6,7,4,3]
    #print(network)
    
    cost_dict = train_network(network, train_full,  num_layers, n_features, 5000, lambd, alpha=0.1)
    test_layer_comps = forward_propagation(network, num_layers, test_full, n_features)
    
    for row, pred in zip(test_layer_comps, test_full):
        #print("Prob array : ", row['A'+str(num_layers-1)])
        prediction = np.argmax(row['A'+str(num_layers-1)])
        actual= np.argmax(pred[n_features:])
        print("Prediction : {} \tActual : {}".format(prediction, actual))
        if prediction==actual:
            correct+=1
        total+=1
        preds.append(prediction)
        actuals.append(actual)
    accuracy, precision, recall, fscore = calculate_accuracy(actuals, preds)
    print("Accuracy in fold-{} is {}".format(i, accuracy))
    print("F1-Score in fold-{} is {}".format(i, fscore))
    accuracy_byfold.append((i, accuracy))
    f1score_byfold.append((i, fscore))


# In[137]:


avg_acc, avg_f1score = 0,0
for i in accuracy_byfold:
    avg_acc += i[1]
avg_acc/len(accuracy_byfold)

for i in f1score_byfold:
    avg_f1score += i[1]
avg_f1score/len(f1score_byfold)


# In[180]:


all_costs = []
x_ax = []
i=5
lambd = 0.01
while i<len(train_full):
    network, num_layers = initialize_network([9,15,20,2])
    cost_dict = train_network(network, train_full[:i],  num_layers, 9, 2000, lambd, alpha=0.3)
    #print(len(train_full[:i]))
    test_layer_comps = forward_propagation(network, num_layers, test_full, 9)
    total_cost = cost_function(test_full, test_layer_comps, network, n_features, lambd)
    all_costs.append(total_cost)
    x_ax.append(i)
    i+=20

plt.plot(x_ax, all_costs) 
plt.xlabel("No. of training instances shown")
plt.ylabel("Cost of network")
plt.title("Cost vs No. of training examples shown")

