import argparse
from helper_fns import *

import os
# import sys
# sys.argv=['']
# del sys
current_directory = os.getcwd()
print("Current working directory:", current_directory)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description ='Run JAW experiments with given tau variation, # trials, mu func, and dataset.')
    
    parser.add_argument('--dataset', type = str, default = 'simulated', help = 'Dataset for experiments.')
    parser.add_argument('--bias', type = float, default = 0.0, help ='Bias for covariate shift')
    parser.add_argument('--sim_data_size', type = int, default = 200, help = 'Only valid for simulated dataset.')
    # parser.add_argument('--muh_fun_name', type = str, default = 'random_forest', help = 'Mu (mean) function predictor.')
    parser.add_argument('--muh_fun_name', type = str, default = 'linear_regression', help = 'Mu (mean) function predictor.')
    # parser.add_argument('--muh_fun_name', type = str, default = 'neural_network', help = 'Mu (mean) function predictor.')
    parser.add_argument('--threshold_type', type = str, default = 'absolute', help = 'Indicator whether threshold is space invariant or not')
    parser.add_argument('--tau', type = float, default = 10.0, help = 'Indicator whether threshold is space invariant or not')
    parser.add_argument('--ntrial', type = int, default = 20, help = 'Number of trials (experiment replicates) to complete.')


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

    if (dataset == 'simulated'):
        # Simulated dataset
        simulated_data = pd.read_csv('0.Datasets/simulated/simulated_data.csv')
        simulated_data = simulated_data[0:sim_data_size]
        X_simulated = simulated_data.iloc[:, 0:2].values # this is all X!!!!!!
        print(X_simulated)
        Y_simulated = simulated_data.iloc[:, 2].values
        print(Y_simulated)
        n_simulated = len(Y_simulated)
        print("X_simulated shape : ", X_simulated.shape)
        
    else:
        raise Exception("Invalid dataset name")

    print("Running dataset " + dataset + ", with muh fun " + muh_fun_name + ", with " + threshold_type + \
              " threshold variation of " + str(tau) +" for ntrial " + str(ntrial))
    
    n_total = eval('n_'+dataset)
    n_train = int(round(0.7*eval('n_'+dataset)))
    
    filler = '/bias_' + str(int(bias)) + '/sim_data_size_' + str(sim_data_size) + '/'
    sigma_eps = 5

    np.random.seed(98765)
    
    if (muh_fun_name == 'gpr'):
        #Standardize data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(eval('X_' + dataset))
        Y_scaled = scaler_y.fit_transform(eval('Y_' + dataset).reshape(-1, 1)).ravel()

        y_scale_factor = scaler_y.scale_
        y_mean = scaler_y.mean_
    
    else:
        X_scaled = eval('X_' + dataset)
        Y_scaled = eval('Y_' + dataset)

### function 1, to genereate training and test         
    X_train, Y_train, X_test, Y_test = generate_data_for_trials(ntrial, n_train, n_total, X_scaled, Y_scaled, bias)


### function 2, to get the cp interval
    # this is like a main function(all method are included)
    Res_all, PDs_all = generate_scores_PD(ntrial, X_train, Y_train, X_test, Y_test, bias, \
                                          muh_fun_name, muh_fun, dataset)
    
    if (muh_fun_name == 'gpr'):
        methods_to_scale = ['CV+_sorted', 'CV+', 'split', 'split_sorted', \
                            'muh_vals_testpoint','muh_split_vals_testpoint', 'any']
        PDs_all[PDs_all['method'] in methods_to_scale].iloc[:,5:] = \
                        PDs_all[PDs_all['method'] in methods_to_scale].iloc[:,5:]*y_scale_factor[0] + y_mean[0]
    
    else:
        None


    Res_all.to_csv(dataset + filler + muh_fun_name + '_' + str(ntrial) + 'Trial'  +'_Res.csv', index = False)
    PDs_all.to_csv(dataset + filler + muh_fun_name + '_' + str(ntrial) + 'Trial'  +'_PDs.csv', index = False)
    
    Res_all = pd.read_csv(dataset + filler + muh_fun_name + '_' + str(ntrial) + 'Trial'  +'_Res.csv')
    PDs_all = pd.read_csv(dataset + filler + muh_fun_name + '_' + str(ntrial) + 'Trial'  +'_PDs.csv')
## this is what?
## use different tau??
    results_by_tau(dataset, filler, muh_fun_name, ntrial, X_test, Y_test, \
                   PDs_all, Res_all, threshold_type, tau, sigma_eps)  
        
    jacknife_plus_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                muh_fun_name + '/JAWS_coverage_by_trial/jackknife_plus.csv', index_col=0)
    jacknife_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                muh_fun_name + '/JAWS_coverage_by_trial/jackknife.csv', index_col=0)
    wt_jackknife_plus_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                muh_fun_name + '/JAWS_coverage_by_trial/wt_jackknife_plus.csv', index_col=0)
    
    cv_plus_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                muh_fun_name + '/JAWS_coverage_by_trial/CV_plus.csv', index_col=0)
    cv_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                muh_fun_name + '/JAWS_coverage_by_trial/CV.csv', index_col=0)
    wt_cv_plus_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                muh_fun_name + '/JAWS_coverage_by_trial/wt_CV_plus.csv', index_col=0)
    
    split_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                muh_fun_name + '/JAWS_coverage_by_trial/split.csv', index_col=0)
    wt_split_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                muh_fun_name + '/JAWS_coverage_by_trial/wt_split.csv', index_col=0)

    
    mean_coverage_by_trial = pd.DataFrame([jacknife_plus_fn.mean(axis = 1).values, \
                                           jacknife_fn.mean(axis = 1).values, \
                                           wt_jackknife_plus_fn.mean(axis = 1).values,\
                                           cv_plus_fn.mean(axis = 1).values, \
                                           cv_fn.mean(axis = 1).values, \
                                           wt_cv_plus_fn.mean(axis = 1).values,\
                                           split_fn.mean(axis = 1).values,
                                           wt_split_fn.mean(axis = 1).values]).T.rename(columns = \
                {0:'Jacknife+', 1:'Jacknife', 2:'Wt_Jacknife+', \
                 3:'CV+', 4:'CV', 5:'Wt_CV+',\
                 6:'Split', 7:'Wt_Split' })
    
    if (bias != 0.0):
        JAWS_methods = ['Jacknife+', 'Wt_Jacknife+', 'CV+', 'Wt_CV+', 'Split', 'Wt_Split']
    else:
        # JAWS_methods = ['Jacknife+', 'CV+', 'Split']
        JAWS_methods = ['Jacknife+',  'Split']
    
    mean_coverage_by_trial = mean_coverage_by_trial[JAWS_methods]
    
    
    
    true_prob_fn = pd.read_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                muh_fun_name + '/JAWS_coverage_by_trial/true_prob.csv', index_col=0)
            
    mean_coverage_by_trial['True Probability'] = true_prob_fn.mean(axis = 1).values 
    
    
    mean_coverage_by_trial.to_csv(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                muh_fun_name + '/JAWS_coverage_by_trial/mean_coverage.csv',index=False)
    
    for i in range(len(mean_coverage_by_trial.columns)):
        plt.hist(mean_coverage_by_trial.iloc[:,i],  alpha = 0.5, label = mean_coverage_by_trial.columns[i] )
    plt.legend(loc = 'upper left')
    plt.savefig(dataset + filler + 'Threshold_type_' + str(threshold_type) + '/' + \
                                muh_fun_name +  '/plots/JAWS_mean_results_' + dataset +'.png')
    plt.clf()
    
    
    
