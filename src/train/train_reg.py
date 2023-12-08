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


## Model options：


def naive(X,Y,X1):
    y_predict = np.repeat(np.mean(Y),len(X1))
    return y_predict

def linear_regression(X,Y,X1):
    """
        Construcat prediction result for given test data set X1
        Input: 
        X: explanatory variable
        Y: response valiable
        X1: explanatory variable, dtypes: <class 'numpy.ndarray'>?
        Output:
        Y1 prediction result for given X1, dtypes: <class 'numpy.ndarray'>?
    """
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

def gpr(X, Y, X1):
    kernel = RationalQuadratic(alpha = 1.0, length_scale = 1.0)
    gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10)
    gp.fit(X,Y)
    return gp.predict(X1)

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



## test with an example


file = './src/data/winequality-red.csv'
df = pd.read_csv(file)




train_dataset = MyDataset(X[:800], y[:800])
val_dataset = MyDataset(X[800:], y[800:])

model = MyModel(n_features=10)
trainer = Trainer(model, train_dataset, val_dataset)
trainer.train()


test_dataset = MyDataset(X[900:], y[900:])  

tester = Tester(model, test_dataset)
tester.test()



X_train, Y_train, X_test, Y_test = generate_data_for_trials(ntrial, n_train, n_total, X_scaled, Y_scaled, bias)
