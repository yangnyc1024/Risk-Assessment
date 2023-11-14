
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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic

from sklearn import svm
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from xgboost import XGBRegressor
from matplotlib.lines import Line2D



## Weighting functions for covariate shift

def sort_both_by_first(v, w):
    zipped_lists = zip(v, w)
    sorted_zipped_lists = sorted(zipped_lists)
    v_sorted = [element for element, _ in sorted_zipped_lists]
    w_sorted = [element for _, element in sorted_zipped_lists]
    
    return [v_sorted, w_sorted]


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


def exponential_tilting_indices(x, bias = 1):
    importance_weights = get_w(x, bias)
    return wsample(importance_weights)

## Generating data and obtainings predictive distribution results

def generate_data_for_trials(ntrial, ntrain, ntotal, X_data, Y_data, bias = 0.0):
   
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


def compute_PDs(X, Y, X1, fit_muh_fun, weights_full, bias):
    ## need to more about this
    
    n = len(Y) ## Num training data
    n1 = X1.shape[0] ## Num test data 

    #################################
    # Naive & jackknife/jack+/jackmm
    #################################

    muh_vals = fit_muh_fun(X,Y,np.r_[X,X1]) 
    # np.r_[X, X1] means that you are creating a new array by appending the array X1 right 
    # after the array X along the first axis (which is usually the row axis in 2-dimensional arrays).
    resids_naive = Y-muh_vals[:n]
    abs_resids_naive = np.abs(resids_naive)
    muh_vals_testpoint = muh_vals[n:]
    
    resids_LOO = np.zeros(n)
    muh_LOO_vals_testpoint = np.zeros((n,n1))
    
    for i in range(n):
        muh_vals_LOO = fit_muh_fun(np.delete(X,i,0),np.delete(Y,i),\
                                   np.r_[X[i].reshape((1,-1)),X1])
        resids_LOO[i] = Y[i] - muh_vals_LOO[0]
        muh_LOO_vals_testpoint[i] = muh_vals_LOO[1:]
    
    abs_resids_LOO = np.abs(resids_LOO)
    
    
    ###############################
    # Weighted jackknife+
    ###############################
    
    # Add infinity
    weights_normalized = np.zeros((n + 1, n1))
    sum_train_weights = np.sum(weights_full[0:n])
    for i in range(0, n + 1):
        for j in range(0, n1):
            if (i < n):
                weights_normalized[i, j] = weights_full[i] / (sum_train_weights + weights_full[n + j])
            else:
                weights_normalized[i, j] = weights_full[n+j] / (sum_train_weights + weights_full[n + j])
    
        
#     ###############################
#     # CV+
#     ###############################

    K = 10
    n_K = np.floor(n/K).astype(int)
    base_inds_to_delete = np.arange(n_K).astype(int)
    resids_LKO = np.zeros(n)
    muh_LKO_vals_testpoint = np.zeros((n,n1))
    for i in range(K):
        inds_to_delete = (base_inds_to_delete + n_K*i).astype(int)
        muh_vals_LKO = fit_muh_fun(np.delete(X,inds_to_delete,0),np.delete(Y,inds_to_delete),\
                                   np.r_[X[inds_to_delete],X1])
        resids_LKO[inds_to_delete] = Y[inds_to_delete] - muh_vals_LKO[:n_K]
        for inner_K in range(n_K):
            muh_LKO_vals_testpoint[inds_to_delete[inner_K]] = muh_vals_LKO[n_K:]
    
    abs_resids_LKO = np.abs(resids_LKO)


#     ###############################
#     # split conformal
#     ###############################
    
    idx = np.random.permutation(n)
    n_half = int(np.floor(n/2))
    idx_train, idx_cal = idx[:n_half], idx[n_half:]
    muh_split_vals = fit_muh_fun(X[idx_train],Y[idx_train],np.r_[X[idx_cal],X1])
    resids_split = Y[idx_cal] - muh_split_vals[:(n-n_half)]
    muh_split_vals_testpoint = muh_split_vals[(n-n_half):]
    
    abs_resids_split = np.abs(resids_split)

    ###############################
    # weighted split conformal
    ###############################
    
    ## Add infty (distribution on augmented real line)
    positive_infinity = np.array([float('inf')])
    unweighted_split_vals = np.concatenate([resids_split, positive_infinity])
    
#     ## Get normalized weights:
    
#     wsplit_quantiles = np.zeros(n1)   
        
    weights_normalized_wsplit = np.zeros((n_half + 1, n1))
    sum_cal_weights = np.sum(weights_full[idx_cal])
    for j in range(0, n1):
        for i in range(0, n_half + 1):
            if (i < n_half):
                i_cal = idx_cal[i]
                weights_normalized_wsplit[i, j] = weights_full[i_cal] / (sum_cal_weights + weights_full[n + j])
            else:
                weights_normalized_wsplit[i, j] = weights_full[n+j] / (sum_cal_weights + weights_full[n + j])
            

    #################################
    # construct prediction intervals
    #################################
    
    col_names = np.concatenate((['lower' + str(i) for i in range(0, n1)], ['upper' + str(i) for i in range(0, n1)]))
    
    Abs_Res_dict = { 
                'jackknife' : pd.DataFrame(np.sort(abs_resids_LOO)),
                'CV' : pd.DataFrame(np.sort(abs_resids_LKO)),
                'split' : pd.DataFrame(np.sort(abs_resids_split))} 
        
        
    PDs_dict = {
                'jackknife' : pd.DataFrame(\
                    np.c_[np.sort(np.tile(muh_vals_testpoint, (n, 1)).T - np.tile(abs_resids_LOO, (n1, 1))).T, \
                        np.sort(np.tile(muh_vals_testpoint, (n, 1)).T + np.tile(abs_resids_LOO, (n1, 1))).T],\
                           columns = col_names),\
                'jackknife+_sorted' : pd.DataFrame(\
                    np.c_[np.sort(muh_LOO_vals_testpoint.T - abs_resids_LOO,axis=1).T, \
                        np.sort(muh_LOO_vals_testpoint.T + abs_resids_LOO,axis=1).T],\
                           columns = col_names),\
                'jackknife+' : pd.DataFrame(\
                    np.c_[(muh_LOO_vals_testpoint.T - abs_resids_LOO).T, \
                        (muh_LOO_vals_testpoint.T + abs_resids_LOO).T],\
                           columns = col_names),\
                'CV+_sorted' : pd.DataFrame(\
                    np.c_[np.sort(muh_LKO_vals_testpoint.T - abs_resids_LKO,axis=1).T, \
                        np.sort(muh_LKO_vals_testpoint.T + abs_resids_LKO,axis=1).T],\
                           columns = col_names),\
                'CV+' : pd.DataFrame(\
                    np.c_[(muh_LKO_vals_testpoint.T - abs_resids_LKO).T, \
                        (muh_LKO_vals_testpoint.T + abs_resids_LKO).T],\
                           columns = col_names),\
                'split' : pd.DataFrame(\
                    np.c_[np.sort(np.tile(muh_split_vals_testpoint, (n_half, 1)).T - np.tile(abs_resids_split, (n1, 1))).T, \
                           np.sort(np.tile(muh_split_vals_testpoint, (n_half, 1)).T + np.tile(abs_resids_split, (n1, 1))).T],\
                            columns = col_names),\
                'split_sorted' : pd.DataFrame(\
                    np.c_[np.sort(np.tile(muh_split_vals_testpoint, (n_half, 1)).T - np.tile(np.sort(abs_resids_split), (n1, 1))).T, \
                           np.sort(np.tile(muh_split_vals_testpoint, (n_half, 1)).T + np.tile(np.sort(abs_resids_split), (n1, 1))).T],\
                            columns = col_names),\
               'weights_split_test' : pd.DataFrame(\
                    np.concatenate((weights_normalized_wsplit[n_half, :], weights_normalized_wsplit[n_half, :])).reshape((1, 2*n1)),\
                           columns = col_names),\
               'weights_JAW_test' : pd.DataFrame(\
                    np.concatenate((weights_normalized[n, :], weights_normalized[n, :])).reshape((1, 2*n1)),\
                           columns = col_names),\
                'weights_split_train' : pd.DataFrame(\
                    np.c_[weights_normalized_wsplit[0:n_half, :], weights_normalized_wsplit[0:n_half, :]],\
                           columns = col_names),\
                'weights_JAW_train' : pd.DataFrame(\
                    np.c_[weights_normalized[0:n, :], weights_normalized[0:n, :]],\
                           columns = col_names),\
               'muh_vals_testpoint' : pd.DataFrame(\
                    np.concatenate((muh_vals_testpoint, muh_vals_testpoint)).reshape((1, 2*n1)),\
                           columns = col_names),\
               'muh_split_vals_testpoint' : pd.DataFrame(\
                    np.concatenate((muh_split_vals_testpoint, muh_split_vals_testpoint)).reshape((1, 2*n1)),\
                           columns = col_names)}
    
                
    return Abs_Res_dict, PDs_dict



def generate_scores_PD(ntrial, X_by_trial , Y_by_trial, X1_by_trial, Y1_by_trial, bias, muh_fun_name, muh_fun, dataset = 'wine'):
    # this is to generate predict interval
    PDs_method_names = ['jackknife', 'jackknife+_sorted', 'jackknife+', 'CV+_sorted', 'CV+', 'split', 'split_sorted',\
                        'muh_vals_testpoint','muh_split_vals_testpoint', 'weights_split_train', 'weights_JAW_train', 'weights_split_test', 'weights_JAW_test']
    Res_method_names = ['jackknife', 'CV' ,'split']
    
    ntest = len(Y1_by_trial[0])
    ntrain = len(Y_by_trial[0])
    
    PDs_col_names = np.concatenate((['itrial','dataset','muh_fun','method','testpoint'], \
                                    ['lower' + str(i) for i in range(0, ntest)], ['upper' + str(i) for i in range(0, ntest)]))
    PDs_all = pd.DataFrame(columns = PDs_col_names)
    
    Res_col_names = ['itrial','dataset','muh_fun','method','testpoint','value']
    Res_all = pd.DataFrame(columns = Res_col_names)

    for itrial in tqdm.tqdm(range(ntrial)):
        
        X = X_by_trial[itrial]
        Y = Y_by_trial[itrial]
        X1 = X1_by_trial[itrial]
        Y1 = Y1_by_trial[itrial]
        
        X_full = np.concatenate((X, X1), axis = 0)
        
        if (bias != 0.0):
            weights_full = get_w(X_full, bias).reshape(len(X_full))
        else: 
            weights_full = np.ones(len(X_full))
        ## what is muh_fun here?
        Res, PDs = compute_PDs(X, Y, X1, muh_fun, weights_full, bias)
            
        for method in PDs_method_names:
            if (method in ['weights_split_test', 'weights_JAW_test', 'muh_split_vals_testpoint', 'muh_vals_testpoint']):
                info = pd.DataFrame([itrial,dataset,muh_fun_name,method,False]).T
                info.columns = ['itrial','dataset','muh_fun','method','testpoint']
                
            elif (method in ['weights_split_train', 'split', 'split_sorted']):
                n_half = int(np.floor(ntrain/2))
                info = pd.DataFrame(np.tile([itrial,dataset,muh_fun_name,method,False], (n_half, 1)), \
                                columns = ['itrial','dataset','muh_fun','method','testpoint'])
            else:
                info = pd.DataFrame(np.tile([itrial,dataset,muh_fun_name,method,False], (ntrain, 1)), \
                                columns = ['itrial','dataset','muh_fun','method','testpoint'])
            info_PD_method_results = pd.concat([info, PDs[method]], axis = 1, ignore_index=True)
            info_PD_method_results.reset_index(drop = True, inplace = True)
            info_PD_method_results.columns = PDs_all.columns
            PDs_all.reset_index(drop = True, inplace = True)
            PDs_all = pd.concat([PDs_all, info_PD_method_results], ignore_index = True, axis = 0)
            PDs_all.reset_index(drop = True, inplace = True)
        
        test_point_row = \
        pd.DataFrame(np.concatenate(([itrial,dataset,muh_fun_name,'any',True],Y1.squeeze(),Y1.squeeze()))).T
        test_point_row.columns = PDs_all.columns
        test_point_row.reset_index(drop = True, inplace = True)
        PDs_all.reset_index(drop = True, inplace = True)
        PDs_all = pd.concat([PDs_all, test_point_row], ignore_index = True, axis = 0)
        PDs_all.reset_index(drop = True, inplace = True)
        
                
        for method in Res_method_names:
            if (method == 'split'):
                n_half = int(np.floor(ntrain/2))
                info = pd.DataFrame(np.tile([itrial,dataset,muh_fun_name,method,False], (n_half, 1)), \
                                columns = ['itrial','dataset','muh_fun','method','testpoint'])
            else:
                info = pd.DataFrame(np.tile([itrial,dataset,muh_fun_name,method,False], (ntrain, 1)), \
                                columns = ['itrial','dataset','muh_fun','method','testpoint'])
            info_res_method_results = pd.concat([info, Res[method]], axis = 1, ignore_index = True)
            info_res_method_results.reset_index(drop = True, inplace = True)
            info_res_method_results.columns = Res_all.columns
            Res_all.reset_index(drop = True, inplace = True)
            Res_all = pd.concat([Res_all, info_res_method_results], ignore_index = True, axis = 0)
            Res_all.reset_index(drop = True, inplace = True)

    return Res_all, PDs_all



def generate_true_probs(ntrial, X1_by_trial, Y1_by_trial, PDs_data, threshold_type, tau_to_use, sigma_eps):
    
    prob_true = []

    for itrial in tqdm.tqdm(range(ntrial)):
        
        x_test = X1_by_trial[itrial]
        x1_test = x_test.T[0]
        x2_test = x_test.T[1]
        
        y_test = Y1_by_trial[itrial]
        y_pred = PDs_data[PDs_data['method'] == 'muh_vals_testpoint'][PDs_data['itrial'] == itrial].iloc[:,5:5+len(y_test)].values[0]
        
        diff_pred_true = y_pred - (x1_test*np.abs(np.log(np.abs(x2_test/100))) + \
                                   x2_test*np.abs(np.log(np.abs(x1_test/100))))
        
        if threshold_type == 'relative':
            tau_test_pt = np.abs(y_pred)*tau_to_use
        else:
            tau_test_pt = tau_to_use
        
        prob_true.append(scipy.stats.norm(0, sigma_eps).cdf(diff_pred_true + tau_test_pt) - \
                    scipy.stats.norm(0, sigma_eps).cdf(diff_pred_true - tau_test_pt)) 
        
    return prob_true
