# -*- coding: utf-8 -*-
"""
@author: rraaj
"""

import time
import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.neural_network import MLPClassifier

print("START")

# datasets
print("Loading trainig data...")
train_data = pd.read_csv("datasets/fashion-mnist_train.csv").values
print("Loading testing data...")
test_data = pd.read_csv("datasets/fashion-mnist_test.csv").values
print("Loading complete.")

# training data
train_X = train_data[0:20000,1:]
train_y = train_data[0:20000,0]

# testing data
test_X = test_data[0:,1:]
test_y = test_data[0:,0]

# parameters
looped = False
solver = 'adam' # solvers ('adam', 'sgd', or 'lbfgs')
alpha_man = 0.0001 # learning rate
layer_man = 100 # neurons in hidden layers
alpha_table = [0.1, 0.01, 0.001, 0.0001]
layer_table = []
for i in range(10,310,10):
    layer_table.append(i)
times = [[],[],[],[]]
scores = [[],[],[],[]]
row = 0

# single/loop
if looped == True:
    for alpha in alpha_table:
        for layer in layer_table:
            # Multi-Layer Perceptron
            clf = MLPClassifier(solver=solver,alpha=0.0001,
                                learning_rate_init=alpha,
                                hidden_layer_sizes=(layer, ),
                                random_state=None)
            
            # parameters
            print("----------------------------------------")
            print("alpha: ", alpha, ", layers: ", layer)
            
            # training
            print("Training MLP...")
            start = time.time()
            clf.fit(train_X,train_y)
            end = time.time()
            
            # runtime
            runtime = end - start
            print ("Runtime = ", runtime, "seconds")
            times[row].append(runtime)
            
            # accuracy
            accuracy = 100*clf.score(test_X,test_y)
            print("Accutacy = ", accuracy, "%")
            scores[row].append(accuracy)
            
        row = row+1
        
    
else:
    # Multi-Layer Perceptron
    clf = MLPClassifier(solver=solver,alpha=0.0001,
                        learning_rate_init=alpha_man,
                        hidden_layer_sizes=(layer_man, ),
                        random_state=None)
    
    # parameters
    print("----------------------------------------")
    print("learning rate: ", alpha_man, ", layers: ", layer_man)
    
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
    test_elem = 1000
    d = test_X[test_elem]
    d.shape = (28,28)
    pt.imshow(255-d,cmap='gray')
    print("predicted value: ", clf.predict( [test_X[test_elem]] ))
    pt.show()
    print("actual class: ", test_y[test_elem])
    
    # accuracy
    print("Accutacy = ", 100*clf.score(test_X,test_y), "%")


print("STOP")
