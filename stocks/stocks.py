# -*- coding: utf-8 -*-
"""
@author: rraaj
"""

#all: 1259
#dataset: 10 - 1249

import time
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from random import random
print("START")

# data
print("Loading data...")
data = pd.read_csv("datasets/stocks/MSFT_data.csv").values

# dataset
print("Creating dataset...")
data_X1 = data[9:1249,1:5]
data_X2 = data[8:1248,1:5]
data_X3 = data[7:1247,1:5]
data_X4 = data[6:1246,1:5]
data_X5 = data[5:1245,1:5]
data_X6 = data[4:1244,1:5]
data_X = np.append(data_X1,data_X2,axis=1)
data_X = np.append(data_X,data_X3,axis=1)
data_X = np.append(data_X,data_X4,axis=1)
data_X = np.append(data_X,data_X5,axis=1)
data_X = np.append(data_X,data_X6,axis=1)
data_y = data[10:1250,1]

#train-test data
print("Splitting dataset...")
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y,
                                                    test_size=0.33,
                                                    random_state=0)


# parameters
looped = False
solver = 'lbfgs'
alpha = 0.0001
learning_rate_man = 0.0001
layers_man = 100
learning_rate_table = [0.1, 0.01, 0.001, 0.0001]
layers_table = []
for i in range(10,310,10):
    layers_table.append(i)
times = [[],[],[],[]]
scores = [[],[],[],[]]
av_time = 0
av_error= 0
row = 0



# single/loop

if looped == True:
    for learning_rate in learning_rate_table:
        for layers in layers_table:
            av_error = 0
            av_time = 0
            # parameters
            print("----------------------------------------")
            print("learnin rate: ", learning_rate, ", layers: ", layers)
            print("Training MLP...")            
            for i in range(0,100):
                #train-test data
                train_X, test_X, train_y, test_y = train_test_split(data_X, data_y,
                                                                    test_size=0.33,
                                                                    random_state=i)
                # Multi-Layer Perceptron Regressor
                clf = MLPRegressor(solver=solver,alpha=alpha,
                                    learning_rate_init=learning_rate,
                                    hidden_layer_sizes=(layers, ),
                                    random_state=None)
                
                # training
                start = time.time()
                clf.fit(train_X,train_y)
                end = time.time()
                
                # runtime
                runtime = end - start
                av_time = av_time + runtime
                
                # accuracy
                relative_error = 100 - 100*clf.score(test_X,test_y)
                av_error = av_error + relative_error
                
            av_error = av_error/100
            print("Runtime = ",av_time, "seconds")
            print("Relative error = ",av_error,"%")
            av_time = av_time/100
            times[row].append(av_time)
            scores[row].append(av_error)
            
        row = row+1
    
else:
    #train-test data
    print("Splitting dataset...")
    train_X, test_X, train_y, test_y = train_test_split(data_X, data_y,
                                                    test_size=0.33,
                                                    random_state=1)
    # Multi-Layer Perceptron Regressor
    clf = MLPRegressor(solver=solver,alpha=alpha,
                        learning_rate_init=learning_rate_man,
                        hidden_layer_sizes=(layers_man, ),
                        random_state=None)
    
    # parameters
    print("----------------------------------------")
    print("learning rate: ", learning_rate_man, ", layers: ", layers_man)
    
    # training
    print("Training MLP...")
    start = time.time()
    clf.fit(train_X,train_y)
    end = time.time()
    
    # runtime
    runtime = end - start
    print ("Runtime = ", runtime, "seconds")
    
    # testing
    
    # Single image test
    test_elem = 200
    noise_amp = 5
    print("actual value: ", test_y[test_elem])
    print("W/OUT NOISE")
    print("predicted value: ", clf.predict( [test_X[test_elem]] )[0])
    relative_error = 100*(clf.predict( [test_X[test_elem]]) - test_y[test_elem])/test_y[test_elem]
    print("Relative error: ", relative_error[0], "%")
    
    for i in range(0,24):
        test_X[test_elem][i] = test_X[test_elem][i] + noise_amp*(random() - 0.5)
    
    print("W/ NOISE")
    print("predicted value: ", clf.predict( [test_X[test_elem]] )[0])

    
    # accuracy
    relative_error = 100*(clf.predict( [test_X[test_elem]]) - test_y[test_elem])/test_y[test_elem]
    print("Relative error: ", relative_error[0], "%")


print("STOP")