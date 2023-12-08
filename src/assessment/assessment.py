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

def prob_by_residuals(Res_itrial, tau_test_pt, method):
    ## calculate the pobability?
    idx = 0
    scores = list(Res_itrial[Res_itrial['method']==method]['value'])
    n = len(scores)
    while (idx < n and scores[idx] < tau_test_pt):
        idx += 1
    if (idx == n or (idx > 0 and scores[idx] >= tau_test_pt)):
        idx -= 1 
    
    return idx / (n + 1)





def prob_by_pred_dists(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, method):
    ## calculate the pobability?
    ## it is for one trial but n test point
    ## Find lower point

    idx_low = 0
    train_scores_lower = list(PDs_itrial[PDs_itrial['method'] == method]['lower' + str(test_pt)])
    n = len(train_scores_lower)
    while (idx_low < n and train_scores_lower[idx_low] < y_pred_lower):
        idx_low += 1

    ## Find upper point
    idx_up = 0
    train_scores_upper = list(PDs_itrial[PDs_itrial['method'] == method]['upper' + str(test_pt)])
    while (idx_up < n and train_scores_upper[idx_up] < y_pred_upper):
        idx_up += 1
    if (idx_up == n or (idx_up > 0 and train_scores_upper[idx_up] >= y_pred_upper)):
        idx_up -= 1
    
    alpha1 = idx_low / (n + 1)
    alpha2 = 1 - (idx_up / (n + 1))
    return min(1 - alpha1, 1 - alpha2)

file = 'src/assessment/linear_regression_10Trial_PDs.csv'
test = pd.read_csv(file)
result = prob_by_pred_dists(test, 0, 100, 0, 'linear_regression')
print(f"this test result is {result}")

def prob_interval_JAW(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, method):
    ## calculate the pobability?
    if (method == 'jackknife+' or method == 'CV+'):
        weights_to_use = 'weights_JAW_'
    elif (method == 'split'):
        weights_to_use = 'weights_split_'
    else:
        print('Error')
    
    weights = list(PDs_itrial[PDs_itrial['method'] == str(weights_to_use + 'train')]['lower' + str(test_pt)])
    
    train_scores_lower = list(PDs_itrial[PDs_itrial['method'] == method ]['lower' + str(test_pt)])
    train_scores_upper = list(PDs_itrial[PDs_itrial['method'] == method ]['upper' + str(test_pt)])
    
    n = len(train_scores_lower) - 1

    ### Add infty
    weights.append(float(PDs_itrial[PDs_itrial['method'] == str(weights_to_use + 'test')]['lower' + str(test_pt)]))
    positive_infinity = float('inf')
    train_scores_lower.append(-positive_infinity)
    train_scores_upper.append(positive_infinity)

    train_scores_lower_sorted, weights_lower_sorted = sort_both_by_first(train_scores_lower, weights)
    train_scores_upper_sorted, weights_upper_sorted = sort_both_by_first(train_scores_upper, weights)
    

    ## Find lower point
    ## Want low_weight to equal sum of all weights less than a_L + weight of smallest point greater than a_L
    idx_low = 0
    low_weight = weights_lower_sorted[idx_low]
    while (idx_low <= n and train_scores_lower_sorted[idx_low] < y_pred_lower):
        idx_low += 1 
        low_weight += weights_lower_sorted[idx_low]
        
    ## Find upper point
    idx_up = 0
    up_weight = 0
    while (idx_up <= n and train_scores_upper_sorted[idx_up] < y_pred_upper):
        up_weight += weights_upper_sorted[idx_up]
        idx_up += 1 ## This is id of next one whose weight hasn't been added yet
        
    if (idx_up == n+1 or (idx_up > 0 and train_scores_upper[idx_up] >= y_pred_upper)):
        idx_up -= 1
        up_weight -= weights_upper_sorted[idx_up]
    
    beta1 = low_weight ## alpha_E on lower values
    beta2 = up_weight ## 1 - alpha_E on upper values
    
    return min(1 - beta1, beta2)



    
# def generate_prob_results_by_tau(method, PDs_itrial, Res_itrial, threshold_type, tau_to_use):
#     """
#         calculate probability for given predict interval & true probability
#         Input: 
#             method: cp method
#             PDs_itrial: dataframe with predict interval
#             Res_itrial: dataframe with true probability
#             threshold_type: type = str, default = 'absolute'& 'relative', help = 'Indicator whether threshold is space invariant or not'
#             tau_to_use: ??? type = float, default = 10.0, help = 'Indicator whether threshold is space invariant or not')
#         Output:
#         Y1 prediction result for given X1, dtypes: <class 'numpy.ndarray'>?
#     """
#     ## what is the n_test here?
#     n_test = int(PDs_itrial.columns[-1].split('upper')[1]) + 1
#     probs = []

#     for test_pt in range(0, n_test):

#         y_true = float(PDs_itrial[PDs_itrial['testpoint'] == True]['lower' + str(test_pt)])

#         if (method == 'wt_split'):
#             y_pred = float(PDs_itrial[PDs_itrial['method'] == 'muh_split_vals_testpoint']['lower' + str(test_pt)])
#         else:
#             y_pred = float(PDs_itrial[PDs_itrial['method'] == 'muh_vals_testpoint']['lower' + str(test_pt)])

#         if threshold_type == 'relative':
#             tau_test_pt = np.abs(y_pred)*tau_to_use
#         else:
#             tau_test_pt = tau_to_use

#         y_pred_lower = y_pred - tau_test_pt
#         y_pred_upper = y_pred + tau_test_pt

#         if (method in ['jackknife', 'CV', 'split']):
#             probs.append(prob_by_residuals(Res_itrial, tau_test_pt, method))
#         elif (method == 'jackknife_plus'):
#             probs.append(prob_by_pred_dists(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, 'jackknife+_sorted'))
#         elif (method == 'wt_jackknife_plus'):
#             probs.append(prob_interval_JAW(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, 'jackknife+'))
#         elif (method == 'CV_plus'):
#             probs.append(prob_by_pred_dists(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, 'CV+_sorted'))
#         elif (method == 'wt_CV_plus'):
#             probs.append(prob_interval_JAW(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, 'CV+'))
#         elif (method == 'wt_split'):
#             probs.append(prob_interval_JAW(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, 'split'))
#         else:
#             print('Error - method not specified')

#     return probs


# def results_by_tau(dataset_to_use, filler, muh_fun_name, ntrial, X1_by_trial, Y1_by_trial, PDs_data, Res_data, \
#                    threshold_type, tau_to_use, sigma_eps = None):
# ## the main probability calculation function
#     """
#         Construct prediction result for given test data set X1
#         Input: 
#             dataset_to_use: here is the dataset , input like ???
#             filler:
#             muh_fun_name: machine learning method name
#             ntrial: how many trials? what is this for?
#             threshold_type: 
#                     for different calculate probabilty method?
#                     'absolute', help = 'Indicator whether threshold is space invariant or not'
#                     'relevant'
#             tau_to_use: 
#                     type = float, default = 10.0, help = 'Indicator whether threshold is space invariant or not')
#             PDs_data: with prediction interval data set
#             Res_data: with true probability data set
#         Output:
#             Null, only to generate the probability into file? which file????
#     """

#     if (dataset_to_use == 'simulated'):
#         prob_true = generate_true_probs(ntrial, X1_by_trial, Y1_by_trial, PDs_data, \
#                                         threshold_type, tau_to_use, sigma_eps)
#         pd.DataFrame(prob_true).to_csv(dataset_to_use + filler + 'Threshold_type_' + str(threshold_type) \
#                                 + '/' + muh_fun_name + '/JAWS_coverage_by_trial/true_prob.csv')
        
#     prob_methods = ['jackknife_plus', 'jackknife', 'wt_jackknife_plus', 'CV_plus',  'CV', 'wt_CV_plus', 'split', 'wt_split']
    
#     jackknife_plus_probs = []
#     jackknife_probs = []
#     wt_jackknife_plus_probs = []
    
#     CV_plus_probs = []
#     CV_probs = []
#     wt_CV_plus_probs = []
    
#     split_probs = []
#     wt_split_probs = []
    
#     for j in tqdm.tqdm(range(ntrial)):
#         PDs_itrial = PDs_data[PDs_data['itrial'] == j]
#         Res_itrial = Res_data[Res_data['itrial'] == j]
# ## main function to calculate the probability, named generate_prob_results_by_tau       
#         for i in range(len(prob_methods)):
#             eval(prob_methods[i] + '_probs').append(generate_prob_results_by_tau(prob_methods[i], PDs_itrial, Res_itrial, threshold_type, tau_to_use))
    
#     for i in range(len(prob_methods)):
#         pd.DataFrame(eval(prob_methods[i] + '_probs')).to_csv(dataset_to_use + filler + 'Threshold_type_' + str(threshold_type) + '/' +\
#                                           muh_fun_name + '/JAWS_coverage_by_trial/' + str(prob_methods[i]) + '.csv')

#     return None

## function 3, calculate risk assessment probability stage: to get the probability
# results_by_tau(dataset, filler, muh_fun_name, ntrial, X_test, Y_test, \
#                    PDs_all, Res_all, threshold_type, tau, sigma_eps)  