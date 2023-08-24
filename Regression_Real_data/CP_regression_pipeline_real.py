import argparse
from helper_fns import *

import os
# import sys
# sys.argv=['']
# del sys


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description ='Run JAW experiments with given tau variation, # trials, mu func, and dataset.')
    
    parser.add_argument('--dataset', type = str, default = 'simulated', help = 'Dataset for experiments.')
    parser.add_argument('--bias', type = float, default = 0.0, help ='Bias for covariate shift')
    parser.add_argument('--sim_data_size', type = int, default = 1000, help = 'Only valid for simulated dataset.')
    parser.add_argument('--muh_fun_name', type = str, default = 'random_forest', help = 'Mu (mean) function predictor.')
    parser.add_argument('--threshold_type', type = str, default = 'absolute', help = 'Indicator whether threshold is space invariant or not')
    parser.add_argument('--tau', type = float, default = 10.0, help ='Indicator whether threshold is space invariant or not')
    parser.add_argument('--ntrial', type = int, default = 100, help ='Number of trials (experiment replicates) to complete.')

    args = parser.parse_args()
    dataset = args.dataset
    bias = args.bias
    sim_data_size = args.sim_data_size
    muh_fun_name = args.muh_fun_name
    threshold_type = args.threshold_type
    tau = args.tau
    ntrial = args.ntrial
    
    muh_fun = eval(muh_fun_name)  
    
    warnings.filterwarnings('ignore', message = 'Boolean Series key will be reindexed to match DataFrame index.')

    if (dataset == 'airfoil'):
        airfoil = pd.read_csv('0.Datasets/airfoil/airfoil.txt', sep = '\t', header=None)
        airfoil.columns = ["Frequency","Angle","Chord","Velocity","Suction","Sound"]
        #airfoil = airfoil[0:sim_data_size]
        X_airfoil = airfoil.iloc[:, 0:5].values
        X_airfoil[:, 0] = np.log(X_airfoil[:, 0])
        X_airfoil[:, 4] = np.log(X_airfoil[:, 4])
        Y_airfoil = airfoil.iloc[:, 5].values
        n_airfoil = len(Y_airfoil)
        print("X_airfoil shape : ", X_airfoil.shape)
        
    elif (dataset == 'wine'):
        winequality_red = pd.read_csv('0.Datasets/wine/winequality-red.csv', sep=';')
        X_wine = winequality_red.iloc[:, 0:11].values
        Y_wine = winequality_red.iloc[:, 11].values
        n_wine = len(Y_wine)
        print("X_wine shape : ", X_wine.shape)
        
    elif (dataset == 'wave'):
        wave = pd.read_csv('0.Datasets/WECs_DataSet/Adelaide_Data.csv', header = None)
        X_wave = wave.iloc[0:2000, 0:48].values
        Y_wave = wave.iloc[0:2000, 48].values
        n_wave = len(Y_wave)
        print("X_wave shape : ", X_wave.shape)
        
    elif (dataset == 'superconduct'):
        superconduct = pd.read_csv('0.Datasets/superconduct/train.csv')
        X_superconduct = superconduct.iloc[0:2000, 0:81].values
        Y_superconduct = superconduct.iloc[0:2000, 81].values
        n_superconduct = len(Y_superconduct)
        print("X_superconduct shape : ", X_superconduct.shape)
        
    else:
        raise Exception("Invalid dataset name")
    
    
    print("Running dataset " + dataset + ", with muh fun " + muh_fun_name + ", with " + threshold_type + \
          " threshold variation of " + str(tau) +" for ntrial " + str(ntrial))
    
    n_total = eval('n_'+dataset)
    n_train = int(round(0.7*eval('n_'+dataset)))
    
    filler = '/'
    filler2 = '/tau_' + str(int(tau)) + '/'
    sigma_eps = None

    np.random.seed(98765)
    

    X_train, Y_train, X_test, Y_test = generate_data_for_trials(ntrial, n_train, n_total, eval('X_' + dataset), eval('Y_' + dataset), bias)
    
#     Res_all, PDs_all = generate_scores_PD(ntrial, X_train, Y_train, X_test, Y_test, bias, muh_fun_name, muh_fun, dataset)

#     Res_all.to_csv(dataset + filler + muh_fun_name + '_' + str(ntrial) + 'Trial'  +'_Res.csv', index = False)
#     PDs_all.to_csv(dataset + filler + muh_fun_name + '_' + str(ntrial) + 'Trial'  +'_PDs.csv', index = False)
    
    Res_all = pd.read_csv(dataset + filler + muh_fun_name + '_' + str(ntrial) + 'Trial'  +'_Res.csv')
    PDs_all = pd.read_csv(dataset + filler + muh_fun_name + '_' + str(ntrial) + 'Trial'  +'_PDs.csv')
    
    results_by_tau(dataset, filler, filler2, muh_fun_name, ntrial, X_test, Y_test, \
                   PDs_all, Res_all, threshold_type, tau, sigma_eps)

        
    jacknife_plus_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + filler2 + \
                                muh_fun_name + '/JAWS_coverage_by_trial/jackknife_plus.csv', index_col=0)
    jacknife_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + filler2 + \
                                muh_fun_name + '/JAWS_coverage_by_trial/jackknife.csv', index_col=0)
    
    cv_plus_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + filler2 + \
                                muh_fun_name + '/JAWS_coverage_by_trial/CV_plus.csv', index_col=0)
    cv_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + filler2 + \
                                muh_fun_name + '/JAWS_coverage_by_trial/CV.csv', index_col=0)
    
    split_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + filler2 + \
                                muh_fun_name + '/JAWS_coverage_by_trial/split.csv', index_col=0)

    
    mean_coverage_by_trial = pd.DataFrame([jacknife_plus_fn.mean(axis = 1).values, \
                                           jacknife_fn.mean(axis = 1).values, \
                                           cv_plus_fn.mean(axis = 1).values, \
                                           cv_fn.mean(axis = 1).values, \
                                           split_fn.mean(axis = 1).values,\
                                           empirical_prop(PDs_all, n_total - n_train, tau)\
                                          ]).T.rename(columns = \
                {0:'Jacknife+', 1:'Jacknife', 2:'CV+', 3:'CV', 4:'Split', 5 : 'Empirical'})
    
    
    
    JAWS_methods = ['Jacknife+', 'CV+', 'Split', 'Empirical']
    mean_coverage_by_trial = mean_coverage_by_trial[JAWS_methods]
    
    
    mean_coverage_by_trial.to_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + filler2 + \
                                muh_fun_name + '/JAWS_coverage_by_trial/mean_coverage.csv',index=False)
    
    for i in range(len(mean_coverage_by_trial.columns)):
        plt.hist(mean_coverage_by_trial.iloc[:,i],  alpha = 0.5, label = mean_coverage_by_trial.columns[i] )
    plt.legend(loc = 'upper left')
    plt.savefig(dataset + filler + 'Threshold_type_' + str(threshold_type) + filler2 + \
                                muh_fun_name +  '/plots/JAWS_mean_results_' + dataset +'.png')
    plt.clf()
    
    
    
