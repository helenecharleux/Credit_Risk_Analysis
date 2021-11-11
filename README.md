# Credit_Risk_Analysis

## Project overview
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. The purpose of this project is to test different machine learning models. Thanks to our analysis of the imbalanced classification reports, we will be able to highlight if one of those models could be used to lower the risk associated with credit.

## Machine Learning Model analysis

### 1. Resampling models analysis
We have first used the oversampling RandomOverSampler, SMOTE algorithms, and the undersampling ClusterCentroids algorithm. 

The oversampling RandomOverSampler and SMOTE algorithms have shown the same results. Threy have a balanced accuracy score of 66,6% and a confusion matrix as below:


Predictions | Predicted True | Predicted False
----------------|------------ | -------------
Actually True | 71 | 30
Actually False | 6,334 | 10,770

Those two models have a low precision rate. However they have a good sensitivity because they are able to identify a large proportion of the credit which are at risk. 

The undersampling model has a lower accuracy score (54,4%) and a lower precision rate. The confusion matrix is presented below:

Predictions | Predicted True | Predicted False
----------------|------------ | -------------
Actually True | 70 | 31
Actually False | 10,340 | 6,764

The imbalanced classification record supports our conclusion: there is a pronounced imbalance between the sensitivity and the precision rate of those models. The F1 score for those three models is low. We will only present one of the imbalance classification record in order to support our analysis. The following report is for the undersampling model.

![Imbalanced_matrix](https://user-images.githubusercontent.com/85641189/141006223-9c6d9b2a-116f-4bf8-bd0d-4a4138db6d41.png)

### 2. SMOTEEN: combinatorial approach of over- and undersampling



