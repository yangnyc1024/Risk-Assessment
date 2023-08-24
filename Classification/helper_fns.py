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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
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


def naive(X,Y,X1):
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
    return np.array(model(X1)).flatten()


def generate_data_for_trials(ntrial, ntrain, ntotal):
    
    train_inds = list(range(ntrial))
    test_inds = list(range(ntrial))
    
    for itrial in range(ntrial):
        
        np.random.seed(itrial)
        train_inds[itrial] = np.random.choice(ntotal, ntrain, replace = False)
        test_inds[itrial] = np.setdiff1d(np.arange(ntotal), train_inds[itrial])
    
    return train_inds, test_inds


def compute_PDs(X,Y,X1,fit_muh_fun):
    #print("Computing predictive distributions")
    n = len(Y) ## Num training data
    
    n1 = X1.shape[0]
    
    ################################
    # Naive & jackknife/jack+/jackmm
    ################################
    
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
    # CV+
    ###############################

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
    resids_split = Y[idx_cal]-muh_split_vals[:(n-n_half)]
    abs_resids_split = np.abs(resids_split)
    muh_split_vals_testpoint = muh_split_vals[(n-n_half):]

    ##############################################
    # construct prediction intervals and residuals 
    ##############################################
    
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
                'jackknife+' : pd.DataFrame(\
                    np.c_[np.sort(muh_LOO_vals_testpoint.T - abs_resids_LOO,axis=1).T, \
                        np.sort(muh_LOO_vals_testpoint.T + abs_resids_LOO,axis=1).T],\
                           columns = col_names),\
                'CV' : pd.DataFrame(\
                    np.c_[np.sort(np.tile(muh_vals_testpoint, (n, 1)).T - np.tile(abs_resids_LKO, (n1, 1))).T, \
                        np.sort(np.tile(muh_vals_testpoint, (n, 1)).T + np.tile(abs_resids_LKO, (n1, 1))).T],\
                           columns = col_names),\
                'CV+' : pd.DataFrame(\
                     np.c_[np.sort(muh_LKO_vals_testpoint.T - abs_resids_LKO,axis=1).T, \
                         np.sort(muh_LKO_vals_testpoint.T + abs_resids_LKO,axis=1).T],\
                            columns = col_names),\
                'split' : pd.DataFrame(\
                     np.c_[np.sort(np.tile(muh_split_vals_testpoint, (n_half, 1)).T - np.tile(abs_resids_split, (n1, 1))).T, \
                            np.sort(np.tile(muh_split_vals_testpoint, (n_half, 1)).T + np.tile(abs_resids_split, (n1, 1))).T],\
                             columns = col_names),\
                'muh_vals_testpoint' : pd.DataFrame(\
                    np.concatenate((muh_vals_testpoint, muh_vals_testpoint)).reshape((1, 2*n1)),\
                           columns = col_names)}  
                
    return Abs_Res_dict,  PDs_dict


def generate_scores_PD(ntrial, muh_fun_name, muh_fun, X_data, Y_data, train_inds, test_inds, dataset = 'wine'):
    PDs_method_names = ['jackknife+', 'jackknife', 'CV+', 'CV', 'split', 'muh_vals_testpoint']
    Res_method_names = ['jackknife', 'CV' ,'split']
    
    n1_ = len(test_inds[0]) #Size of test data
    ntrain = len(train_inds[0])
    
    PDs_col_names = np.concatenate((['itrial','dataset','muh_fun','method','testpoint'], \
                                    ['lower' + str(i) for i in range(0, n1_)], ['upper' + str(i) for i in range(0, n1_)]))
    PDs_all = pd.DataFrame(columns = PDs_col_names)
    
    Res_col_names = ['itrial','dataset','muh_fun','method','testpoint','value']
    Res_all = pd.DataFrame(columns = Res_col_names)

    for itrial in tqdm.tqdm(range(ntrial)):

        X = X_data[train_inds[itrial]]
        Y = Y_data[train_inds[itrial]]
        X1 = X_data[test_inds[itrial]]
        Y1 = Y_data[test_inds[itrial]]
        
        Res, PDs = compute_PDs(X, Y, X1, muh_fun)
        
        for method in PDs_method_names:
            if(method in ['muh_vals_testpoint']):
                info = pd.DataFrame([itrial,dataset,muh_fun_name,method,False]).T
                info.columns = ['itrial','dataset','muh_fun','method','testpoint']
            elif (method == 'split'):
                n_half = int(np.floor(ntrain/2))
                info = pd.DataFrame(np.tile([itrial,dataset,muh_fun_name,method,False], (n_half, 1)), \
                                columns = ['itrial','dataset','muh_fun','method','testpoint'])
            else:
                info = pd.DataFrame(np.tile([itrial,dataset,muh_fun_name,method,False], (ntrain, 1)), \
                                columns = ['itrial','dataset','muh_fun','method','testpoint'])
            info_PD_method_results = pd.concat([info, PDs[method]], axis = 1, ignore_index=True)
            info_PD_method_results.reset_index(drop=True, inplace=True)
            info_PD_method_results.columns = PDs_all.columns
            PDs_all.reset_index(drop=True, inplace=True)
            PDs_all = pd.concat([PDs_all, info_PD_method_results], ignore_index=True, axis=0)
            PDs_all.reset_index(drop=True, inplace=True)
        
        test_point_row = \
        pd.DataFrame(np.concatenate(([itrial,dataset,muh_fun_name,'any',True],Y1.squeeze(),Y1.squeeze()))).T
        test_point_row.columns = PDs_all.columns
        test_point_row.reset_index(drop=True, inplace=True)
        PDs_all.reset_index(drop=True, inplace=True)
        PDs_all = pd.concat([PDs_all, test_point_row], ignore_index=True, axis=0)
        PDs_all.reset_index(drop=True, inplace=True)
        
                
        for method in Res_method_names:
            if (method == 'split'):
                n_half = int(np.floor(ntrain/2))
                info = pd.DataFrame(np.tile([itrial,dataset,muh_fun_name,method,False], (n_half, 1)), \
                                columns = ['itrial','dataset','muh_fun','method','testpoint'])
            else:
                info = pd.DataFrame(np.tile([itrial,dataset,muh_fun_name,method,False], (ntrain, 1)), \
                                columns = ['itrial','dataset','muh_fun','method','testpoint'])
            info_res_method_results = pd.concat([info, Res[method]], axis = 1, ignore_index=True)
            info_res_method_results.reset_index(drop=True, inplace=True)
            info_res_method_results.columns = Res_all.columns
            Res_all.reset_index(drop=True, inplace=True)
            Res_all = pd.concat([Res_all, info_res_method_results], ignore_index=True, axis=0)
            Res_all.reset_index(drop=True, inplace=True)

    return Res_all, PDs_all
    
    
def generate_true_probs(ntrial, X_data, Y_data, PDs_data, train_inds, test_inds, threshold_type, tau_to_use, sigma_eps):
    
    prob_true = []

    for itrial in tqdm.tqdm(range(ntrial)):
        
        x_test = X_data[test_inds[itrial]]
        x1_test = x_test.T[0]
        x2_test = x_test.T[1]
        
        y_test = Y_data[test_inds[itrial]]
        y_pred = PDs_data[PDs_data['method']=='muh_vals_testpoint'][PDs_data['itrial'] == itrial].iloc[:,5:5+len(test_inds[0])].values[0]
        
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
    train_scores_lower = list(PDs_itrial[PDs_itrial['method']==method]['lower' + str(test_pt)])
    n = len(train_scores_lower)
    while (idx_low < n and train_scores_lower[idx_low] < y_pred_lower):
        idx_low += 1

    ## Find upper point
    idx_up = 0
    train_scores_upper = list(PDs_itrial[PDs_itrial['method']==method]['upper' + str(test_pt)])
    while (idx_up < n and train_scores_upper[idx_up] < y_pred_upper):
        idx_up += 1
    if (idx_up == n or (idx_up > 0 and train_scores_upper[idx_up] >= y_pred_upper)):
        idx_up -= 1
    
    alpha1 = idx_low / (n + 1)
    alpha2 = 1 - (idx_up / (n + 1))
    
    return min(1 - alpha1, 1 - alpha2)


def generate_prob_results_by_tau(PDs_data, Res_data, threshold_type, tau_to_use, n_test):
    
    jackknife_plus_probs_ALL = []
    jackknife_probs_ALL = []
    CV_plus_probs_ALL = []
    CV_probs_ALL = []
    split_probs_ALL = []
    correctness_ALL = []

    for itrial in tqdm.tqdm(set(PDs_data['itrial'])):
        PDs_itrial = PDs_data[PDs_data['itrial']==itrial] 
        Res_itrial = Res_data[Res_data['itrial']==itrial]
        
        jackknife_plus_probs_itrial = []
        jackknife_probs_itrial = []
        CV_plus_probs_itrial = []
        CV_probs_itrial = []
        split_probs_itrial = []
        correctness_itrial = []

        for test_pt in range(0, n_test):
            y_true = float(PDs_itrial[PDs_itrial['testpoint']==True]['lower' + str(test_pt)])
            y_pred = float(PDs_itrial[PDs_itrial['method']=='muh_vals_testpoint']['lower' + str(test_pt)])
            
            if threshold_type == 'relative':
                tau_test_pt = np.abs(y_pred)*tau_to_use
            else:
                tau_test_pt = tau_to_use
                
            y_pred_lower = y_pred - tau_test_pt
            y_pred_upper = y_pred + tau_test_pt

            jackknife_plus_probs_itrial.append(prob_by_pred_dists(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, 'jackknife+'))
            jackknife_probs_itrial.append(prob_by_residuals(Res_itrial, tau_test_pt, 'jackknife'))
            CV_plus_probs_itrial.append(prob_by_pred_dists(PDs_itrial, y_pred_lower, y_pred_upper, test_pt, 'CV+'))
            CV_probs_itrial.append(prob_by_residuals(Res_itrial, tau_test_pt, 'CV'))
            split_probs_itrial.append(prob_by_residuals(Res_itrial, tau_test_pt, 'split'))
            correctness_itrial.append(y_pred_lower <= y_true and y_true <= y_pred_upper)

        jackknife_plus_probs_ALL.append(jackknife_plus_probs_itrial)
        jackknife_probs_ALL.append(jackknife_probs_itrial)
        CV_plus_probs_ALL.append(CV_plus_probs_itrial)
        CV_probs_ALL.append(CV_probs_itrial)
        split_probs_ALL.append(split_probs_itrial)
        correctness_ALL.append(correctness_itrial)
        
    return jackknife_plus_probs_ALL, jackknife_probs_ALL, CV_plus_probs_ALL, CV_probs_ALL, split_probs_ALL, correctness_ALL


def results_by_tau(dataset_to_use, filler, muh_fun_name, ntrial, X_data, Y_data, PDs_data, Res_data, \
                   train_inds, test_inds, threshold_type, tau_to_use, sigma_eps = None):
    
    if dataset_to_use == 'simulated':
        prob_true = generate_true_probs(ntrial, X_data, Y_data, PDs_data, train_inds, test_inds, \
                                        threshold_type, tau_to_use, sigma_eps)
        true_prob_fn = []
        
    jackknife_plus_probs, jackknife_probs, cv_plus_probs, cv_probs, split_probs, correctness =\
    generate_prob_results_by_tau(PDs_data, Res_data, threshold_type, tau_to_use, len(test_inds[0]))
    
    jacknife_plus_fn = []
    jacknife_fn = []
    cv_plus_fn = []
    cv_fn = []
    split_fn = []
    correctness_fn = []
    
    for itrial in range(ntrial):
        
        jackknife_plus_probs_trial = jackknife_plus_probs[itrial]
        jackknife_probs_trial = jackknife_probs[itrial]
        cv_plus_probs_trial = cv_plus_probs[itrial]
        cv_probs_trial = cv_probs[itrial]
        split_probs_trial = split_probs[itrial]
        correctness_trial = correctness[itrial]
        
        jacknife_plus_fn.append(jackknife_plus_probs_trial)
        jacknife_fn.append(jackknife_probs_trial)
        cv_plus_fn.append(cv_plus_probs_trial)
        cv_fn.append(cv_probs_trial)
        split_fn.append(split_probs_trial)
        correctness_fn.append(correctness_trial)
        
        if dataset_to_use == 'simulated':
            prob_true_trial = prob_true[itrial]
            true_prob_fn.append(prob_true_trial)
    
            pd.DataFrame(true_prob_fn).to_csv(dataset_to_use + filler + '/Threshold_type_' + str(threshold_type) \
                                + '/' + muh_fun_name + '/JAWS_coverage_by_trial/true_prob_fn_by_trial.csv')
    
    pd.DataFrame(jacknife_plus_fn).to_csv(dataset_to_use + filler + '/Threshold_type_' + str(threshold_type) + '/' +\
                                          muh_fun_name + '/JAWS_coverage_by_trial/jacknife_plus_fn_by_trial.csv')
    pd.DataFrame(jacknife_fn).to_csv(dataset_to_use + filler + '/Threshold_type_' + str(threshold_type) + '/' + \
                                          muh_fun_name +  '/JAWS_coverage_by_trial/jacknife_fn_by_trial.csv')
    pd.DataFrame(cv_plus_fn).to_csv(dataset_to_use + filler + '/Threshold_type_' + str(threshold_type) + '/' + \
                                          muh_fun_name + '/JAWS_coverage_by_trial/cv_plus_fn_by_trial.csv')
    pd.DataFrame(cv_fn).to_csv(dataset_to_use + filler + '/Threshold_type_' + str(threshold_type) + '/' + \
                                          muh_fun_name +  '/JAWS_coverage_by_trial/cv_fn_by_trial.csv')
    pd.DataFrame(split_fn).to_csv(dataset_to_use + filler + '/Threshold_type_' + str(threshold_type) + '/' + \
                                          muh_fun_name + '/JAWS_coverage_by_trial/split_fn_by_trial.csv')
    pd.DataFrame(correctness_fn).to_csv(dataset_to_use + filler + '/Threshold_type_' + str(threshold_type) + '/' + \
                                          muh_fun_name + '/JAWS_coverage_by_trial/correctness_fn_by_trial.csv')

    return None


def plot_interval_sequence(dataset, model, Y_test, Y_pred, threshold_type = 'absolute', tau = 10, max_count = 100):
    
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
    plt.figure(figsize=(optimal_width, 4))
    ax = plt.gca() 
        
    valid_interval = (Y_test < test_upper_limit) & (Y_test > test_lower_limit)
    colors = np.array(['#e67e22', '#27ae60'])[valid_interval[:max_count].astype(int)]
    
    
    for i in range(max_count):
        ax.plot([i, i], [test_lower_limit[i], test_upper_limit[i]], c='#3498db')   
        ax.plot([i, i], [test_lower_limit[i], test_upper_limit[i]+1], '_',c='#3498db')
  
    ax.scatter(range(max_count), Y_test[:max_count], marker='x', zorder=3, color=colors)
    
    # Plot the observed samples
    ax.set_ylabel('Y', fontsize=14)
    ax.set_xlabel('Test point index', fontsize=14)
    ax.legend(['Acceptable prediction band for Y'])
    ax.tick_params(axis = 'both', which = 'major', labelsize=14)
    
    plt.title('True value vs '+ model + ' prediction')
    plt.savefig(dataset + '/sim_data_size_' + str(1000) + '/Threshold_type_' + str(threshold_type) + '/' + \
                                    model + '/plots/prediction_intervals_' + dataset +'.png')
    
    return ax 


def coverage_by_model(cp_types, sim_data_size, threshold_type, model, model_names):
    
    dataset = 'simulated'
    
    mean_coverage_by_model = []
    for i in range(len(model)):
        mean_coverage_by_model.append(pd.read_csv(dataset + '/sim_data_size_' + str(sim_data_size) +\
                                    '/' + '/Threshold_type_' + str(threshold_type) + '/' + str(model[i]) + \
                                    '/JAWS_coverage_by_trial/mean_coverage_by_trial.csv'))
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
                hue = 'Legend')
    plt.legend(loc = 'upper left')
    plt.ylabel('Coverage')
    plt.xlabel('Model')
    plt.title('Coverage estimate distribution by  model')
    
    for i in range(len(model)):
        plt.savefig(dataset + '/sim_data_size_1000/Threshold_type_' + str(threshold_type) + '/' + \
                                    model[i] + '/plots/coverage_estimates_by_model_' + dataset +'.png')
    #plt.clf()
    
    return None


def coverage_by_dataset_size(cp_types, sim_data_size, threshold_type, model):
    
    dataset = 'simulated'
    
    mean_coverage_by_dataset_size = []
    for i in range(len(sim_data_size)):
        mean_coverage_by_dataset_size.append(pd.read_csv(dataset + '/sim_data_size_' + str(sim_data_size[i]) +\
                                    '/' + '/Threshold_type_' + str(threshold_type) + '/' + model + \
                                    '/JAWS_coverage_by_trial/mean_coverage_by_trial.csv'))
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
    plt.figure(figsize=(12,6))
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
    

def rep_alpha_results(cp_types, dataset, sim_data_size, threshold_type, muh_fun_name):
    
    if dataset == 'simulated':
        filler = '/sim_data_size_' + str(sim_data_size) + '/' 
    else:
        filler = '/'

    jacknife_plus_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                muh_fun_name + '/JAWS_coverage_by_trial/jacknife_plus_fn_by_trial.csv', index_col=0)
    cv_plus_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                    muh_fun_name +'/JAWS_coverage_by_trial/cv_plus_fn_by_trial.csv', index_col=0)
    split_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                    muh_fun_name + '/JAWS_coverage_by_trial/split_fn_by_trial.csv', index_col=0)
    
    # if dataset == 'simulated':
    #     true_prob_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
    #                             muh_fun_name + '/JAWS_coverage_by_trial/true_prob_fn_by_trial.csv', index_col=0)
    #     data_collate = [jacknife_plus_fn, cv_plus_fn, split_fn, true_prob_fn]
    # else:
    
    data_collate = [jacknife_plus_fn, cv_plus_fn, split_fn]
    
    rows = len(data_collate)
    cols = 4
    temp = np.array([0.0]*rows*cols).reshape(rows,cols)

    for i in range(len(data_collate)):
        temp[i] = [data_collate[i].mean(axis = 1).mean(), data_collate[i].mean(axis = 1).min(), \
                   data_collate[i].min(axis = 1).mean(),data_collate[i].min(axis = 1).min() ]  

    rep_alpha_by_cp_type = pd.DataFrame(temp).rename(columns = \
                   {0:'Overall Avg', 1:'Min (Avg)', 2:'Avg (Min)', 3:'Overall Min'})
    rep_alpha_by_cp_type['Index'] = cp_types[0:rows]
    rep_alpha_by_cp_type = rep_alpha_by_cp_type.set_index('Index')
    
    rep_alpha_by_cp_type.style.to_latex(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                muh_fun_name + '/tables/rep_alpha_' + dataset +'.tex')

    return rep_alpha_by_cp_type
