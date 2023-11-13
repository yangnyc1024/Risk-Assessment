# In doc: CP_regression_pipeline_simulated

## To do list

- Function: generate_scores_PD is training？or it is for predefine interval?

  - ​        \## what is muh_fun here? in generate_scores_PD, the parameter pass into the compute_PDS
  - muh_fun: I think it is the algorithm we choose to use

  



## New Design Idea

- Training(training folder)
- generate interval？(cp method folder)
- generate probability (invert cp method folder)
- experiment
  - graph function






## Function: generate_data_for_trials

- parameter:
  - training data set, bias

- design for:
  - random split the data into two set
  - this could work with bias result

- output: 
  - split data

## Function: generate_scores_PD(generate interval)



### Function: compute_PDS(different cp method for interval)

- parameters:
  - dataset, method, weight?, bias(covariate shift)
  - training data set(X, Y), testing data set(X_1,Y_1)
- design for:
  - **compuate a dictionary???????????**
  - construct predict interval???
  - Yes this is for generating interval for given data set
  - **this includes all different methods**





## Some unknown function





## Function: results_by_tau(generate score)

#### Function：generate_true_probs



#### Function：generate_prob_results_by_tau（CORE!!）： help find probability

- use different prob_by_residuals to calculate probability
- it includes all generate_prob
  - prob_by_residuals
  - prob_by_pred_dists
  - prob_interval_JAW





## post processing

- put all probability into mean_coverage_by_trial
- get related true_probability_fn
- put the plot?