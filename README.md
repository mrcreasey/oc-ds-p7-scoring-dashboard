# Project 7 - Implement a scoring model

## Problem

A financial company offers consumer credit for people with little or no loan history, and wishes to
implement **a “credit scoring” tool** to decide whether to accept or refuse credit.

This project aims to:

- Develop **a scoring model** to predict the probability of payment default for these customers,
  based on mostly financial data sources
- Develop **an interactive dashboard** for customer relationship managers to explain credit granting
  decisions as transparently as possibleS

## Motivation

This is project 7 for the **Master in Data Science** (in French, BAC+5) from OpenClassrooms.

The project demonstrates separation of concerns: model code, API and dashboard:

- **code** : Handling **imbalanced data** for a binary classification model
- **API** : Creation of an application programming interface to serve the saved model (to any number
  of dashboards)
- **dashboard** : Visualisation of the data from the api: predicted scores and their interpretation

## Requirements

**Data** : The dataset (~700Mb) and descriptions can be downloaded from
<https://www.kaggle.com/c/home-credit-default-risk/data>. It consists of financial data of 307511
anonymized customers, provided in seven tables, with a target column 'TARGET' informing if the
client repaid his loan (0) or was in default (1)

**Python libraries** : This project is composed of 3 phases :

- the modelling **code** : data integration, cleaning and creation of the classification model
- the scoring model **api** : a backend for serving model predictions
- the interactive **dashboard** : a frontend for visualising model scores and their interpretation
  for a selected client

The python requirements for each phase are similar (see requirements.txt), but not identical.

- **code** :
  `imbalanced-learn, numpy, pandas, matplotlib, seaborn, scikit-learn, lightgbm, yellowbrick, shap`
- **api** : `flask, gunicorn, numpy, pandas, scikit-learn, lightgbm, shap`
- **dashboard** : `streamlit, ipython, pandas, matplotlib, scikit-learn, lightgbm, shap`

For maintenance and reduced deployment dependencies, each of these 3 phases should have their own
requirements.txt, in separate version controlled git submodules.

## Files

_Notes_ : Files are in French. Open https://nbviewer.org/ and paste notebook GitHub url if GitHub
takes too long to render.\_

The main files are:

- [code/P7_eda_nettoyage.ipynb](./code/P7_eda_nettoyage.ipynb): Exploratory Data Analysis (EDA) and
  data cleaning notebook (joining and aggregating data from 8 tables).
  <a href="https://nbviewer.org/github/mrcreasey/oc-ds-p7-scoring-dashboard/blob/main/code/P7_eda_nettoyage.ipynb" target="blank"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="Open in nb-viewer"/></a>
- [code/P7_modelisation.ipynb](./code/P7_modelisation.ipynb): Development of the credit scoring
  model, handling imbalanced data and using a custom scoring threshold.
  <a href="https://nbviewer.org/github/mrcreasey/oc-ds-p7-scoring-dashboard/blob/main//code/P7_modelisation.ipynb" target="blank"><img src="https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg" alt="Open in nb-viewer"/></a>

- [Note Méthodologique.pdf](./Note+Méthodologique.pdf) : Model training methodology, business cost
  function, evaluation metric, global and local interpretability

- [P7_presentation.pdf](./P7_presentation.pdf): Presentation slides

## Folders for dashboard

Code for the model, API and dashboard are in the "**code**", **"api"** and **"dashboard"** folders
respectively.

## Data Exploration and Modelling

### Data cleaning, simple feature engineering and data merging

The financial tables (loan request, repayment history, previous loans, external data) were merged by
JOIN on the customer key (SK_ID_CURR), with [a script](https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features/script) that had already produced good classification
results, with a few adaptations, resulting in 602 numeric columns for 307507 customers

### Imbalanced data

The data has already been [explored](https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction/notebook) in [detail](https://www.kaggle.com/gpreda/home-credit-default-risk-extensive-eda) during a kaggle competition:

This exploration shows that:
- the distribution of **the target is very unbalanced**:
- less than 8% of customers are in default.

If we make a prediction that all customers are good, we will have an accuracy of 93% for the
majority class, but we will have identified no defaulting customers.

### Train-Test Split and Preprocessing

The cleaned dataset was divided between the train (80%) and test (20%) datasets. A pre-processing
pipeline was set up to avoid data leakage. Missing values were replaced by the median value (all
columns are already numeric). For feature selection and modeling, where needed, the data was scaled
with StandardScaler.

### Feature selection

Most of the 600 columns have very little correlation with the target, and simply add noise to the
model.

To improve modelling time, interpretability and model performance, the top 100
features were selected by a set of feature selection methods
(<https://www.kaggle.com/code/sz8416/6-ways-for-feature-selection>/ ): Filter(KBest,Chi2),
Wrapper(RFE),Embedded(SelectFromModel: LogisticRegression, RandomForest, LightGBM).

Highly collinear columns (VIF > 5) were eliminated
(<https://www.researchgate.net/publication/350525439_Feature_Selection_in_a_Credit_Scoring_Model_Mathematics_ISSN_2227-7390>
)

The final dataset consisted of 79 features for 307507 customers.

### Resampling – target class balancing

For many of the classifiers, the hyperparameter **class_weight = 'balanced'** allows to take into
account imbalances in the target class (cost-sensitive). Several strategies of the imbalanced-learn
library were also tested to rebalance the target classes: Random undersampling (majority class);
Random oversampling (minority class); Synthetic Minority Oversampling Technique (SMOTE); SMOTE
TomekLinks - (majority class under sampling).

### Training via GridSearch with StratifiedKFold cross-validation

To compare the influence of sampling strategy on the performance of models in an acceptable time, a
dataset sample of 10000 was used. Once the sampling strategy, the hyperparameters and the model were
chosen, the final model was trained and optimized on the dataset set. The classifiers tested were:
Dummy (Baseline), RidgeClassifier, LogisticRegression, RandomForest, and LightGBM (Gradient
boosting)

An **imblearn pipeline** allows us to tune the choice of preprocessing, sampling and classifier, to
ensure that cross-validation scores were tested on data without rebalancing.

Several evaluation metrics were calculated: **precision, recall, F1-score, ROC_AUC**, the aim being
to minimize false positives (maximum precision) and false negatives (maximum recall)

### Performance evaluation and choice of best model

The choice of the best model was made by retaining the model with the best ROC_AUC score on the test
set.

The ROC_AUC measures the Area Under the Curve. It shows the trade-off between specificity and
sensitivity (<https://en.wikipedia.org/wiki/Sensitivity_and_specificity>)

The closer the curve approaches the upper left corner, the better the specificity and sensitivity
(and therefore precision and recall)

For decision tree methods, the SMOTE seems to have the effect of overfitting on the training game,
because on the test game we see a significant drop in predictive ability.

The Light LGBM model without resampling, but with parameters {class_weight = balanced, max_depth=6}
is the best performing (high ROC_AUC score on the test data, faster to compute), and therefore is
chosen as the best model.

## The business cost function, optimization algorithm and evaluation metric

### The business cost function

For the bank, the cost of providing a loan to a customer who does not repay his loan (false negative
(FN)-type II error) is more than the loss of refusing a loan to a customer who will not have loan
problems (false positive (FP) – type I error).

- **Recall** = `TP/(TP+FN)` : maximise recall == minimise the false negatives
- **Precision** = `TP/(TP+FP)` : maximiser precision == minimise les false positives
- **F1 score** is a balance between precision and recall. = `2 * precision * recall / (precision +
  recall)`

To place more weight on recall, we can use the F(beta>1) score: An approximation of the cost for the
bank will be to use F(beta=2) score: 

```f2_score = 5*TP/(5*TP+4*FN + FP)```

A function which estimates the cost for the bank (normalized to stay between 0 and 1, as for the other scorers):

- `profit = (TN `*` value_per_loan_to_good_customer + TP * value_of_refusal_of_loan_to_bad_payer)`
- `loss = (FP * cost_per_loan_refused_to_good_customer + FN * cost_of_giving_a_loan_to_bad_payer)`
- `custom_credit_score = (profit + loss) / (max_profit – max loss)`

Where

- `max_profit = (TN + FP)*tn_profit + (FN+TP)*tp_profit` (give loans only to good payers) ; and
- `max_loss = (TP+FN)*fn_loss + (TN+FP)*fp_loss` (give loans only to bad payers)

For this model, we suppose : tn*profit=1, fp_loss=-0.5, fn_loss=-10, tp_profit=0.2 So,
custom_credit_score = (TN + 0.2*TP - 10*FN - 0.5\*FP) / (max_profit-max_loss)  
3.2 The optimization algorithm The model provides probability values (“pred_proba”) that a customer
will be a good payer (0) and a defaulter (1) • If y_pred = (pred_proba[:,1] > threshold) ==1 (True),
we consider that the customer is defaulting Metrics are calculated by comparing y_pred with true
values (y_true). We retrieve the rate of false positives and negatives from the confusion matrix: •
(TN, FP, FN, TP) = metrics.confusion_matrix(y_test*, y*pred*).ravel()

By changing the discrimination threshold (solvency threshold), we can calculate the business cost
function, to find the optimal threshold for a given business function: For the chosen model, the
optimal threshold is 0.520, coincidentally close to the default threshold = 0.5 We optimize the
model on AUC, then predict ready accepted or refused using the optimal threshold

## The global and local interpretability of the model

### Feature Importance

The model provides the (impurity-based) weights of the model.feature*importances*, based on the
training data.

We can also use sklearn.inspection.permutation_importance to estimate the (entropy-based) feature
importance, based on the permutation of values in each feature of the test data

### SHAP (SHapley Additive exPlanations)

The SHAP method (https://shap.readthedocs.io/ ) calculates the shap_values: the impact of a variable
(on the prediction) for each line of data. SHAP values are additive: values in red increase the
predicted value (risk of failing), a blue value reduces the prediction (risk of failing).

### Global interpretability:

If we take the average of the SHAP values for each feature, we obtain the importance of the features
for the prediction.

We can visualize the distribution of the values of for the most important features via a 'summary
plot', in the form of beeswarm or violin

### Local interpretability:

Negative contributions have an effect of reducing the value of the prediction.

Low risk customer (prob=0.03) ` High risk customer (prob=0.95)



##	Model API (Flask application under Heroku) 
The prediction is made by a Flask application, written in python with the routes:
-	List of client ids: /clients/
-	Customer data: /customer/<id>
-	Prediction (default probability): /predict/<id>
-	Client SHAP explanation: /explain/<id>
-	Global SHAP explanation: /explain/

Deployment is under Heroku at <https://mc-oc-7.herokuapp.com> 

The api source code can be found in the `api` folder (see [api/README.md](./api/README.md) for instructions)

## The interactive dashboard visualization (Streamlit application) 
The dashboard makes requests to the API, because it does not have access to the data or the model.
It is written in python and streamlit, and deployed on share.streamlit.io at the address: 
<https://mrcreasey-oc-ds-p7-scoring-dashboard-dashboardmain-70agjx.streamlitapp.com/>  

The source code for the dashboard can be found in the `dashboard`folder (see [dashboard/README.md](./dashboard/README.md) for instructions)



## Conclusion
### Limitations of the credit scoring model

- The models were calculated on part of the data: it is necessary to analyze the effect of sample
  size on the results (eg via learning curves)
- We cannot completely separate the good payers from the defaulting customers (the ROC_AUC of the
  training data remains between 0.7 and 0.8)
- The application of SMOTE improves training scores, but not validation scores, for the sample size
  used.
- SMOTE quickly becomes too heavy to apply on the entire dataset: the generation of synthetic points
  is very slow, and models created with SMOTE (via imblearn.Pipeline) are too large to be saved
- It is necessary to make the choice between errors of type I (accuracy) and errors of type II
  (recall)
- For the bank, the recall is the most important

## Possible improvements

1. Review feature creation with industry experts: The cleanup, aggregation, merging and feature
   engineering script used seems to have been done without knowledge of the business – many of the
   variables created by the script are irrelevant or duplicated.
2. Review the strategy for dealing with missing values (default median)
3. Improve the selection of features to be adapted to each model (Wrapper/Embedded)
4. Make learning curves to optimize sample size for models
5. Increase the search for the best model hyperparameters
6. Change from Flask API to fastapi (https://fastapi.tiangolo.com/) – faster, automatic request
   documentation, less lines of code, includes authentication and security
7. Add authentication to access the dashboard
8. Added encryption of customer data
9. Store customer data separately from the API (this requires caching it in the API memory,
   otherwise it becomes too slow) – for example in an S3 bucket on AWS
10. Visualize the distribution of each of the most important features for a given client to better
    understand where the client stands among the clients

## Features of this project (keywords)

- Supervised classification, stratified k-fold, cross-validation
- Handling imbalanced data: cost-sensitive, imbalanced-learn, SMOTE, Tomek Links, undersampling, oversampling
- Preprocessing : Filter, embedded, wrapper methods
- Performance metrics: Precision, Recall (sensitivity), specificity, Area Under Curve(AUC), Receiver Operating Characteristic (ROC), ROC_AUC, F1-score, F(beta)-score
- Performance evaluation : Custom cost function, Discrimination threshold
- Interpretability : Permutation importance (impurity-based vs. entropy-based), SHAP(global, local interpretability)
- REST API : Flask, FASTapi, heroku
- Dashboard : Streamlit

## Skills acquired

- Handling imbalanced data
- Using code release software to ensure model integration
- Deployment of a model via an API in the web
- Creation of an interactive dashboard to present model predictions
- Communication of modelling approach in a methodological note

