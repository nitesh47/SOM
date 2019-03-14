#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 16:39:10 2019

@author: nitesh
"""

# Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
sc = MinMaxScaler(feature_range = (0, 1))
# Importing the dataset

class SOM(object):
    def __init__(self,data):
        self.data = data
        
    
    ''' Feature scaling '''
    def scaling(self):
        X = sc.fit_transform(self.data)
        return X

	''' Building SOM'''
    
    def training_som(self):
        som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
        som.random_weights_init(SOM.scaling(self))
        som.train_random(data = SOM.scaling(self), num_iteration = 1)
        return som
        
    ''' Visualizing the results '''
    def Visualize_som(self, X, y):
        bone()
        pcolor(SOM.training_som(self).distance_map().T)
        colorbar()
        markers = ['o', 's']
        colors = ['r', 'g']
        
        for i, x in enumerate(SOM.scaling(self)):
            w = SOM.training_som(self).winner(x)
            plot(w[0] + 0.5,
                 w[1] + 0.5,
                 markers[y[i]],
                 markeredgecolor = colors[y[i]],
                 markerfacecolor = 'None',
                 markersize = 10,
                 markeredgewidth = 2)
        print(show())
    
if __name__ == '__main__':
    
    data = pd.read_csv('Credit_Card_Applications.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    main = SOM(X)
    X_train = SOM.scaling(X)
    
    main.Visualize_som(X_train, y)








