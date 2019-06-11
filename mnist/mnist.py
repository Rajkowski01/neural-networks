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
print("Loading data...")
train_data = pd.read_csv("datasets/mnist_train.csv").values
print("Loading complete.")

# training data
train_X = train_data[0:21000,1:]
train_y = train_data[0:21000,0]

# testing data
test_X = train_data[21000:,1:]
test_y = train_data[21000:,0]

# parameters
looped = True
alpha_man = 0.0001 # alpha
layer_man = 100 # layers
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
            clf = MLPClassifier(solver='adam',alpha=alpha,
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
    clf = MLPClassifier(solver='adam',alpha=alpha_man,
                        hidden_layer_sizes=(layer_man, ),
                        random_state=None)
    
    # parameters
    print("----------------------------------------")
    print("alpha: ", alpha_man, ", layers: ", layer_man)
    
    # training
    print("Training MLP...")
    start = time.time()
    clf.fit(train_X,train_y)
    end = time.time()
    
    # runtime
    runtime = end - start
    print ("Runtime = ", runtime, "seconds")
    
    # testing
    """
    # Single image test
    d = test_X[8]
    d.shape = (28,28)
    pt.imshow(255-d,cmap='gray')
    print("predicted value: ", clf.predict( [test_X[8]] ))
    pt.show()
    """
    
    # accuracy
    print("Accutacy = ", 100*clf.score(test_X,test_y), "%")


print("STOP")
