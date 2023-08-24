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
from sklearn.gaussian_process.kernels import RBF

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


## Model options

def naive(X,Y,X1):
    ## fixed intercept model 
    y_predict = np.repeat(np.mean(Y),len(X1))
    return y_predict

def linear_regression(X,Y,X1):
    reg = LinearRegression().fit(X, Y)
    return reg.predict(X1)

def polynomial_regression(X,Y,X1):
    poly_features = PolynomialFeatures(degree = 5)
    X_poly = poly_features.fit_transform(X)
    X1_poly = poly_features.fit_transform(X1)
    reg = LinearRegression().fit(X_poly,Y)
    return reg.predict(X1_poly)

def knn(X, Y, X1):
    reg = neighbors.KNeighborsRegressor(5)
    return reg.fit(X, Y).predict(X1)

def support_vector(X,Y,X1):
    svr_rbf = svm.SVR(kernel = "rbf", C = 100, gamma = 0.1)
    svr_rbf.fit(X, Y)
    return svr_rbf.predict(X1)

def random_forest(X, Y, X1, ntree = 250, max_depth = 7):
    rf = RandomForestRegressor(n_estimators = ntree, max_depth = max_depth, criterion='absolute_error').fit(X,Y)
    return rf.predict(X1)


def xgb(X, Y, X1, n_est = 500, lr = 0.01, max_depth = 3):
    xgb = XGBRegressor(n_estimators = n_est, learning_rate = lr, max_depth = max_depth)
    xgb.fit(X, Y)
    return xgb.predict(X1)


def neural_network(X, Y, X1, lr = 0.01):
    opt = Adam(learning_rate = lr)
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 300, \
                                    min_delta = 0.001, restore_best_weights=True)

    model = Sequential()
    model.add(Dense(100, activation = 'relu', input_dim = 2))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(1, activation = 'relu'))
    model.compile(optimizer = opt, loss = 'mse')
    # fit model
    model.fit(X, Y, 
              epochs = 2000, 
              validation_split = 0.25,
              callbacks = [early_stopping],
              verbose = 0)
    return np.array(model.predict(X1)).flatten()

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
    
    n = len(Y) ## Num training data
    n1 = X1.shape[0] ## Num test data 

    #################################
    # Naive & jackknife/jack+/jackmm
    #################################

    muh_vals = fit_muh_fun(X,Y,np.r_[X,X1])
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



def prob_by_residuals(Res_itrial, tau_test_pt, method):
    
    idx = 0
    scores = list(Res_itrial[Res_itrial['method']==method]['value'])
    n = len(scores)
    while (idx < n and scores[idx] < tau_test_pt):
        idx += 1
    if (idx == n or (idx > 0 and scores[idx] >= tau_test_pt)):
        idx -= 1 
    
    return idx / (n + 1)



def prob_by_pred_dists(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, method):
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



def prob_interval_JAW(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, method):
    
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
    

    
    
def generate_prob_results_by_tau(method, PDs_itrial, Res_itrial, threshold_type, tau_to_use):
    
    n_test = int(PDs_itrial.columns[-1].split('upper')[1]) + 1
    
    probs = []

    for test_pt in range(0, n_test):

        y_true = float(PDs_itrial[PDs_itrial['testpoint'] == True]['lower' + str(test_pt)])

        if (method == 'wt_split'):
            y_pred = float(PDs_itrial[PDs_itrial['method'] == 'muh_split_vals_testpoint']['lower' + str(test_pt)])
        else:
            y_pred = float(PDs_itrial[PDs_itrial['method'] == 'muh_vals_testpoint']['lower' + str(test_pt)])

        if threshold_type == 'relative':
            tau_test_pt = np.abs(y_pred)*tau_to_use
        else:
            tau_test_pt = tau_to_use

        y_pred_lower = y_pred - tau_test_pt
        y_pred_upper = y_pred + tau_test_pt

        if (method in ['jackknife', 'CV', 'split']):
            probs.append(prob_by_residuals(Res_itrial, tau_test_pt, method))
        elif (method == 'jackknife_plus'):
            probs.append(prob_by_pred_dists(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, 'jackknife+_sorted'))
        elif (method == 'wt_jackknife_plus'):
            probs.append(prob_interval_JAW(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, 'jackknife+'))
        elif (method == 'CV_plus'):
            probs.append(prob_by_pred_dists(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, 'CV+_sorted'))
        elif (method == 'wt_CV_plus'):
            probs.append(prob_interval_JAW(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, 'CV+'))
        elif (method == 'wt_split'):
            probs.append(prob_interval_JAW(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, 'split'))
        else:
            print('Error - method not specified')

    return probs


def results_by_tau(dataset_to_use, filler, filler2, muh_fun_name, ntrial, X1_by_trial, Y1_by_trial, PDs_data, Res_data, \
                   threshold_type, tau_to_use, sigma_eps = None):
    
    if (dataset_to_use == 'simulated'):
        prob_true = generate_true_probs(ntrial, X1_by_trial, Y1_by_trial, PDs_data, \
                                        threshold_type, tau_to_use, sigma_eps)
        pd.DataFrame(prob_true).to_csv(dataset_to_use + filler + 'Threshold_type_' + str(threshold_type) \
                                + '/' + muh_fun_name + '/JAWS_coverage_by_trial/true_prob.csv')
        
        prob_methods = ['jackknife_plus', 'jackknife', 'wt_jackknife_plus', \
                        'CV_plus',  'CV', 'wt_CV_plus', \
                        'split','wt_split']
        
        wt_jackknife_plus_probs = []
        wt_CV_plus_probs = []
        wt_split_probs = []
    
    else:
        prob_methods = ['jackknife_plus', 'jackknife', \
                        'CV_plus',  'CV',  \
                        'split']
        
    
    jackknife_plus_probs = []
    jackknife_probs = []
    
    CV_plus_probs = []
    CV_probs = []
    
    split_probs = []
    
    
    for j in tqdm.tqdm(range(ntrial)):
        PDs_itrial = PDs_data[PDs_data['itrial'] == j]
        Res_itrial = Res_data[Res_data['itrial'] == j]
        
        for i in range(len(prob_methods)):
            eval(prob_methods[i] + '_probs').append(generate_prob_results_by_tau(prob_methods[i], PDs_itrial, Res_itrial, threshold_type, tau_to_use))
    
    for i in range(len(prob_methods)):
        pd.DataFrame(eval(prob_methods[i] + '_probs')).to_csv(dataset_to_use + filler + 'Threshold_type_' \
            + str(threshold_type) + filler2 + muh_fun_name + '/JAWS_coverage_by_trial/' + str(prob_methods[i]) + '.csv')

    return None


def empirical_prop(PDs_all, n_test, tau):
    
    pred_data = PDs_all[PDs_all['method'] == 'muh_vals_testpoint'].iloc[:,5:(5 + \
                                                                n_test)].reset_index(drop = True)
    test_data = PDs_all[PDs_all['testpoint'] == True].iloc[:,5:(5 + n_test)].reset_index(drop = True)
    
    empirical_prop = 1 - np.sum(np.abs(pred_data - test_data) > tau, axis = 1)/n_test
    
    return empirical_prop


## Plotting functions

def plot_interval_sequence(filler, dataset, model, Y_test, Y_pred, threshold_type = 'absolute', tau = 10, max_count = 100):
    
    if threshold_type == 'absolute':
        test_lower_limit = Y_pred - tau
        test_upper_limit = Y_pred + tau
    else:
        test_lower_limit = Y_pred - tau*np.abs(Y_pred)
        test_upper_limit = Y_pred + tau*np.abs(Y_pred)
    

    # Plot at most max_count predictions
    if len(Y_test) <= max_count:
        max_count = len(Y_test)

    optimal_width = max_count / 4
    if optimal_width < 4:
        optimal_width = 4
    plt.figure(figsize = (optimal_width, 4))
    ax = plt.gca() 
        
    valid_interval = (Y_test < test_upper_limit) & (Y_test > test_lower_limit)
    colors = np.array(['#e67e22', '#27ae60'])[valid_interval[:max_count].astype(int)]
    markers = np.array(['^', 'x'])[valid_interval[:max_count].astype(int)]
    
    
    for i in range(max_count):
        ax.plot([i, i], [test_lower_limit[i], test_upper_limit[i]], c = '#3498db')   
        ax.plot([i, i], [test_lower_limit[i], test_upper_limit[i]+1], '_',c = '#3498db')
   
        ax.scatter(range(max_count)[i], Y_test[:max_count][i], marker = markers[i], zorder = 3, color = colors[i])
    
    legend_elements = [Line2D([0], [0],label ='Prediction band for Y'),
                       Line2D([0], [0], marker = 'X', color = 'w', label = 'True Y inside prediction band', markerfacecolor = '#27ae60'),
                       Line2D([0], [0], marker = '^', color = 'w', label = 'True Y outside prediction band', markerfacecolor = '#e67e22')]
    
    # Plot the observed samples
    ax.set_ylabel('Y', fontsize = 14)
    ax.set_xlabel('Test point index', fontsize = 14)
    ax.legend(handles = legend_elements)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 14)
    
    plt.title('True value vs '+ model + ' prediction')
    plt.savefig(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                    model + '/plots/prediction_intervals_' + dataset +'.png')
    
    return ax


def coverage_by_model(dataset, filler, filler2, cp_types, threshold_type, model, model_names, cmap):
    
    mean_coverage_by_model = []
    for i in range(len(model)):
        mean_coverage_by_model.append(pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) \
                                  + filler2 + str(model[i]) + '/JAWS_coverage_by_trial/mean_coverage.csv'))
        mean_coverage_by_model[i]['model'] = np.repeat(model_names[i], 100)

    mean_coverage_by_model_transform = pd.melt(pd.concat([mean_coverage_by_model[0], 
                                                                 mean_coverage_by_model[1], 
                                                                 mean_coverage_by_model[2], 
                                                                 mean_coverage_by_model[3]
                                                         ]
                                                        ),
                                                id_vars = ['model'],
                                                value_vars = cp_types, 
                                                var_name = 'Legend')
    plt.figure(figsize = (12,6))
    sns.boxplot(x = 'model',
                y = 'value',
                data = mean_coverage_by_model_transform,
                hue = 'Legend',
               palette = cmap)
    plt.legend(loc = 'upper left')
    plt.ylabel('Coverage')
    plt.xlabel('Model')
    plt.title('Coverage estimate distribution by  model')
    
    for i in range(len(model)):
        plt.savefig(dataset +  filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                           filler2 + model[i] + '/plots/coverage_estimates_by_model_' + dataset +'.png')
    #plt.clf()
    
    return None


def coverage_by_dataset_size(cp_types, sim_data_size, threshold_type, model):
    
    dataset = 'simulated'
    
    mean_coverage_by_dataset_size = []
    for i in range(len(sim_data_size)):
        mean_coverage_by_dataset_size.append(pd.read_csv(dataset + '/sim_data_size_' + str(sim_data_size[i]) +\
                                    '/' + '/Threshold_type_' + str(threshold_type) + '/' + model + \
                                    '/JAWS_coverage_by_trial/mean_coverage.csv'))
        mean_coverage_by_dataset_size[i]['sim_data_size'] = np.repeat(sim_data_size[i], 100)

    mean_coverage_by_dataset_size_transform = pd.melt(pd.concat([mean_coverage_by_dataset_size[0], 
                                                                 mean_coverage_by_dataset_size[1], 
                                                                 mean_coverage_by_dataset_size[2], 
                                                                 mean_coverage_by_dataset_size[3], 
                                                                 mean_coverage_by_dataset_size[4],
                                                                 mean_coverage_by_dataset_size[5]]),
                                                      id_vars = ['sim_data_size'],
                                                      value_vars = cp_types, 
                                                      var_name = 'Legend')
    plt.figure(figsize = (12,6))
    sns.boxplot(x = 'sim_data_size',
                y = 'value',
                data = mean_coverage_by_dataset_size_transform,
                hue = 'Legend')
    plt.legend(loc = 'upper left')
    plt.ylabel('Coverage')
    plt.xlabel('Size of simulated dataset')
    plt.title('Coverage estimate distribution by size of dataset - ' + model)
    plt.savefig(dataset + '/sim_data_size_' + str(1000) + '/Threshold_type_' + str(threshold_type) + '/' + \
                                    model + '/plots/coverage_estimates_by_size_' + dataset +'.png')
    #plt.clf()
    
    return None
    



