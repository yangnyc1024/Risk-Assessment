# Risk Assessment for regression

## To do list

- figure out stage 2& 3 input or output
- understand Neeraj code training input and output?



## New Design Idea

### src_training

- design for any training function
- input: X(feature), Y()



### src_cPrediction

- 



### src_assessment



- experiment
  - graph function


### experiement





## Previous Design in folder:  CP_regression_pipeline_simulated



### Function: generate_data_for_trials(Stage1: split data)

- parameter:
  - training data set, bias

- design for:
  - random split the data into two set
  - this could work with bias result

- output: 
  - split data

### Function: generate_scores_PD(Stage2: training + generate cp interval)



#### Function: compute_PDS(different cp method for interval)

- parameters:
  - dataset, method, weight?, bias(covariate shift)
  - training data set(X, Y), testing data set(X_1,Y_1)
- design for:
  - **compuate a dictionary???????????**
  - construct predict interval???
  - Yes this is for generating interval for given data set
  - **this includes all different methods**





### Some unknown function





### Function: results_by_tau(Stage 3: generate score)

#### Function：generate_true_probs



#### Function：generate_prob_results_by_tau（CORE!!）： help find probability

- use different prob_by_residuals to calculate probability
- it includes all generate_prob
  - prob_by_residuals
  - prob_by_pred_dists
  - prob_interval_JAW





### post processing

- put all probability into mean_coverage_by_trial
- get related true_probability_fn
- put the plot?