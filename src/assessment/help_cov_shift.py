
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf

import tqdm as tqdm
import random
import warnings
import math
from scipy.stats import norm

from sklearn.preprocessing import StandardScaler


def generate_data_for_trials(ntrial, ntrain, ntotal, X_data, Y_data, bias = 0.0):
    """
        feature engineer main function? 
        to generate trial set (X, Y), and test set (X, Y)
        Input: 
            ntrial: how many trials we would like to repeat?
            ntrain: how many training data points
            ntotal: how many total data points
            X_data: explanatory variable for training, for example, simulated_data.iloc[:, 0:2].values
            Y_data: response valiable for training, for example, simulated_data.iloc[:, 2].values
            bias: related to covariate shift
        Output:
            Y1 prediction result for given X1, dtypes: <class 'numpy.ndarray'>?
    """
   
    X_by_trial = []
    Y_by_trial = []
    X1_by_trial = []
    Y1_by_trial = []
    
    for itrial in range(ntrial):
        
        np.random.seed(itrial)
        
        train_inds = np.random.choice(ntotal, ntrain, replace = False)
        test_inds = np.setdiff1d(np.arange(ntotal), train_inds)
        
        X = X_data[train_inds]
        Y = Y_data[train_inds]
        X1 = X_data[test_inds]
        Y1 = Y_data[test_inds]

        if (bias != 0.0):
            biased_test_indices = exponential_tilting_indices(X1, bias = bias)
            
            X1 = X1[biased_test_indices] 
            Y1 = Y1[biased_test_indices] 
            
        X_by_trial.append(X)
        Y_by_trial.append(Y)
        X1_by_trial.append(X1)
        Y1_by_trial.append(Y1)

    return X_by_trial, Y_by_trial, X1_by_trial, Y1_by_trial




def exponential_tilting_indices(x, bias = 1):
    importance_weights = get_w(x, bias)
    return wsample(importance_weights)


def get_w(x, bias):
    return np.exp(np.log(x) @ [-bias/100, bias/100])# what is the dimensions of x?


def wsample(wts, frac = 0.5):
    
    normalized_wts = wts/max(wts)
    n = len(wts) ## n : length or num of weights
    
    indices = [] ## indices : vector containing indices of the sampled data
    target_num_indices = int(n*frac)
    
    while(len(indices) < target_num_indices): ## Draw samples until have sampled ~25% of samples from D_test
        proposed_indices = np.where(np.random.uniform(size = n) <= normalized_wts)[0].tolist()
        ## If (set of proposed indices that may add is less than or equal to number still needed): then add all of them
        if (len(proposed_indices) <= target_num_indices - len(indices)):
            for j in proposed_indices:
                indices.append(j)
        else: ## Else: Only add the proposed indices that are needed to get to 25% of D_test
            for j in proposed_indices:
                if(len(indices) < target_num_indices):
                    indices.append(j)
    return(indices)



# Simulate data
np.random.seed(0)
n_total = 1000  # Total number of data points
X = np.random.randn(n_total, 2)  # Two explanatory variables
Y = X[:, 0] + 2 * X[:, 1] + np.random.randn(n_total)  # Response variable with some noise


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = Y - np.mean(Y)

ntrial = 5
n_train = 200

# Generate data for trials
X_train, Y_train, X_test, Y_test = generate_data_for_trials(ntrial, n_train, n_total, X_scaled, Y_scaled, bias=0.5)



# Example: Analyze the first trial data
print("First Trial Training Data Shape:", X_train[0].shape, Y_train[0].shape)
print("First Trial Test Data Shape:", X_test[0].shape, Y_test[0].shape)

