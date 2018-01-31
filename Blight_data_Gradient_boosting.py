# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# ## Assignment 4 - Understanding and Predicting Property Maintenance Fines
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. For this assignment, your task is to predict whether a given blight ticket will be paid on time.
# 
# All data for this assignment has been provided to us through the [Detroit Open Data Portal](https://data.detroitmi.gov/). **Only the data already included in your Coursera directory can be used for training the model for this assignment.** Nonetheless, we encourage you to look into data from other Detroit datasets to help inform feature creation and model selection. We recommend taking a look at the following related datasets:
# 
# * [Building Permits](https://data.detroitmi.gov/Property-Parcels/Building-Permits/xw2a-a7tf)
# * [Trades Permits](https://data.detroitmi.gov/Property-Parcels/Trades-Permits/635b-dsgv)
# * [Improve Detroit: Submitted Issues](https://data.detroitmi.gov/Government/Improve-Detroit-Submitted-Issues/fwz3-w3yn)
# * [DPD: Citizen Complaints](https://data.detroitmi.gov/Public-Safety/DPD-Citizen-Complaints-2016/kahe-efs3)
# * [Parcel Map](https://data.detroitmi.gov/Property-Parcels/Parcel-Map/fxkw-udwf)
# 
# ___
# 
# We provide you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible. Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.
# 
# Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.
# 
# <br>
# 
# **File descriptions** (Use only this data for training your model!)
# 
#     train.csv - the training set (all tickets issued 2004-2011)
#     test.csv - the test set (all tickets issued 2012-2016)
#     addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
#      Note: misspelled addresses may be incorrectly geolocated.
# 
# <br>
# 
# **Data fields**
# 
# train.csv & test.csv
# 
#     ticket_id - unique identifier for tickets
#     agency_name - Agency that issued the ticket
#     inspector_name - Name of inspector that issued the ticket
#     violator_name - Name of the person/organization that the ticket was issued to
#     violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
#     mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
#     ticket_issued_date - Date and time the ticket was issued
#     hearing_date - Date and time the violator's hearing was scheduled
#     violation_code, violation_description - Type of violation
#     disposition - Judgment and judgement type
#     fine_amount - Violation fine amount, excluding fees
#     admin_fee - $20 fee assigned to responsible judgments
# state_fee - $10 fee assigned to responsible judgments
#     late_fee - 10% fee assigned to responsible judgments
#     discount_amount - discount applied, if any
#     clean_up_cost - DPW clean-up or graffiti removal cost
#     judgment_amount - Sum of all fines and fees
#     grafitti_status - Flag for graffiti violations
#     
# train.csv only
# 
#     payment_amount - Amount paid, if any
#     payment_date - Date payment was made, if it was received
#     payment_status - Current payment status as of Feb 1 2017
#     balance_due - Fines and fees still owed
#     collection_status - Flag for payments in collections
#     compliance [target variable for prediction] 
#      Null = Not responsible
#      0 = Responsible, non-compliant
#      1 = Responsible, compliant
#     compliance_detail - More information on why each ticket was marked compliant or non-compliant
# 
# 
# ___
# 
# ## Evaluation
# 
# Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
# 
# The evaluation metric for this assignment is the Area Under the ROC Curve (AUC). 
# 
# Your grade will be based on the AUC score computed for your classifier. A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
# ___
# 
# For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using `train.csv`. Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from `test.csv` will be paid, and the index being the ticket_id.
# 
# Example:
# 
#     ticket_id
#        284932    0.531842
#        285362    0.401958
#        285361    0.105928
#        285338    0.018572
#                  ...
#        376499    0.208567
#        376500    0.818759
#        369851    0.018528
#        Name: compliance, dtype: float32
#        
# ### Hints
# 
# * Make sure your code is working before submitting it to the autograder.
# 
# * Print out your result to see whether there is anything weird (e.g., all probabilities are the same).
# 
# * Generally the total runtime should be less than 10 mins. You should NOT use Neural Network related classifiers (e.g., MLPClassifier) in this question. 
# 
# * Try to avoid global variables. If you have other functions besides blight_model, you should move those functions inside the scope of blight_model.
# 
# * Refer to the pinned threads in Week 4's discussion forum when there is something you could not figure it out.

# In[ ]:

import pandas as pd
import numpy as np

#train_merged.iloc[250250:250260,14:16]
#train_merged.loc[train_merged.time_gap <= 0,['ticket_issued_date','hearing_date','time_gap']].head()
#def blight_model():

# Loading Datasets
address = pd.read_csv('addresses.csv')
train = pd.read_csv('train.csv', encoding = 'ISO-8859-1', low_memory = False)
test = pd.read_csv('test.csv')
latlon = pd.read_csv('latlons.csv')

# Filling up null values of co_ordinates
# values corresponding to the null values were taken from google maps
index = [4126,10466,17293,34006,55750,74721,100359]
co_ords = [[42.376728, -83.143197],[42.446590, -83.023177],
         [42.359940, -83.095685],[42.358879, -83.151231],
         [42.358542, -83.080338],[42.383274, -83.058238],
         [42.339996, -83.058551]] 

x , y = list(zip(*co_ords))
x = pd.Series(x,index = index)
y = pd.Series(x,index = index)
for i in index:
    latlon.iloc[i,1] = x[i]
    latlon.iloc[i,2] = y[i]
del x,y,co_ords,index

# Making clusters out of the latitude and longitude values:

from sklearn.cluster import KMeans
# Number of clusters evaluation
#scores = []
#k_range = np.linspace(1,20, num =20 , dtype = int)
#for k in k_range:
#    kmeans = KMeans(n_clusters=k).fit(latlon.loc[:,['lat','lon']])
#    scores.append(kmeans.score(latlon.loc[:,['lat','lon']]))

k = 4
kmeans = KMeans(n_clusters = k, random_state=5100)
#Fitting the data
kmeans.fit(latlon.loc[:,['lat','lon']])

# Clusters
latlon['cluster'] = kmeans.labels_
del i,k

# In[ ]:

# Merging address and latlon datasets
address_merged = pd.merge(address,latlon, how = 'inner',on = 'address')

# Merging test and train with address_merged 
test_merged = pd.merge(test,address_merged,how = 'left',on = 'ticket_id')
train_merged = pd.merge(train,address_merged,how = 'inner', on = 'ticket_id')

# Since we are doing binary classification, we can ignore the null values in the target variable
train_merged = train_merged[pd.notnull(train_merged.compliance)]

## Feature Engineering

# Converting the ticket_issued_date and hearing_date columns to datetime type.
train_merged.ticket_issued_date = pd.to_datetime(train_merged.ticket_issued_date)
train_merged.hearing_date = pd.to_datetime(train_merged.hearing_date)
test_merged.ticket_issued_date = pd.to_datetime(test_merged.ticket_issued_date)
test_merged.hearing_date = pd.to_datetime(test_merged.hearing_date)

# Let's make a new feature which is the time difference between ticket issual and hearing
from datetime import timedelta
train_merged['time_gap'] = (train_merged.hearing_date - train_merged.ticket_issued_date)/timedelta(minutes = 60)
test_merged['time_gap'] = (test_merged.hearing_date - test_merged.ticket_issued_date)/timedelta(minutes = 60)

# Mean time gap
mean_gap = train_merged.loc[(pd.notnull(train_merged.time_gap)) & (train_merged.time_gap > 0),'time_gap'].mean()

# Filling up NaNs and negative values with mean gap
train_merged.loc[(pd.isnull(train_merged.time_gap))|(train_merged.time_gap <= 0), 'time_gap'] = mean_gap
test_merged.loc[(pd.isnull(test_merged.time_gap))|(test_merged.time_gap <= 0),'time_gap'] = mean_gap


# Deleting unnecessary variables to save memory
del train, test,address, latlon,mean_gap

# In[ ]:

## Gradient Boosting Implementation
from sklearn.preprocessing import MinMaxScaler


#variables = ['judgment_amount','late_fee','fine_amount',
#             'discount_amount', 'agency_name','violation_street_name',
#             #'ticket_issued_date','hearing_date',
#             'violation_code','disposition',
#             'time_gap']

# Scaling the numeric features common to both train and test
numeric_variables = ['judgment_amount','late_fee','fine_amount','time_gap',
                     'discount_amount']
scaler = MinMaxScaler()
train_merged.loc[:,numeric_variables] = scaler.fit_transform(train_merged.loc[:,numeric_variables])   
test_merged.loc[:,numeric_variables] = scaler.transform(test_merged.loc[:,numeric_variables])

# Getting dummies for categorical variables
categorical_variables = ['agency_name','violation_code','cluster']#'violation_street_name']
                         

# Converting object to categorical variables
train_merged['compliance'] = train_merged['compliance'].astype('category')

for cat in categorical_variables:
    train_merged[cat] = train_merged[cat].astype('category')
    test_merged[cat] = test_merged[cat].astype('category', categories = list(train_merged[cat].unique()))
del cat



train = pd.DataFrame(index = train_merged.index)
test = pd.DataFrame(index = test_merged.index)
for var in categorical_variables:
    cat_list_train = pd.get_dummies(train_merged[var])
    cat_list_test = pd.get_dummies(test_merged[var])
    train = train.join(cat_list_train)
    test = test.join(cat_list_test)
    del cat_list_test, cat_list_train
del var

#dummy_columns = list(train.columns)
train = train.join(train_merged.loc[:,numeric_variables])
test = test.join(test_merged.loc[:,numeric_variables])

# In[ ]:
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score

#gbm = GradientBoostingClassifier(random_state=5100)
#auc_cross = cross_val_score(gbm,train,train_merged.compliance,cv =5,scoring = 'roc_auc') 

# Fix Learning rate and num
param_test1 = {'n_estimators' : range(20,81,10)}
gsearch1 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate= 0.05,
                                                             min_samples_split=500,
                                                             min_samples_leaf=50,
                                                             max_depth=8,
                                                             max_features='sqrt',
                                                             subsample=0.8,
                                                             random_state = 5100),
    param_grid=param_test1,
    scoring = 'roc_auc',
#    n_jobs = 4,
    cv = 5)
gsearch1.fit(train,train_merged.compliance)

# n_estimators = 80

param_test2 = {'max_depth':range(7,17,2), 'min_samples_split':range(500,2001,500)}
gsearch2 = GridSearchCV(estimator=GradientBoostingClassifier(learning_rate=0.05,
                                                             n_estimators=80,
                                                             max_features='sqrt',
                                                             min_samples_leaf = 50,
                                                             subsample=0.8,
                                                             random_state=5100),
    param_grid=param_test2,
    scoring='roc_auc',
    cv = 5)
gsearch2.fit(train,train_merged.compliance)

# max_depth = 9

param_test3 = {'min_samples_split':[300,400,500], 'min_samples_leaf':range(30,71,10)}
gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.05, 
                                                               n_estimators=80,
                                                               max_depth=9,
                                                               max_features='sqrt', 
                                                               subsample=0.8, 
                                                               random_state=5100),
    param_grid = param_test3, 
    scoring='roc_auc',
    cv=5)

gsearch3.fit(train,train_merged.compliance)

# min_sample_leaf = 30, min_samples_split = 300

param_test4 = {'max_features':range(14,23,2)}
gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1,
                                                               n_estimators=80,
                                                               max_depth=9, 
                                                               min_samples_split=300,
                                                               min_samples_leaf=30, 
                                                               subsample=0.8, 
                                                               random_state=5100),
    param_grid = param_test4, 
    scoring='roc_auc',
    cv=5)
gsearch4.fit(train,train_merged.compliance)

param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.05, 
                                                               n_estimators=80,
                                                               max_depth=9,
                                                               min_samples_split=300, 
                                                               min_samples_leaf=30, 
                                                               random_state=5100,
                                                               max_features=18),
    param_grid = param_test5, 
    scoring='roc_auc',
    cv=5)
gsearch5.fit(train,train_merged.compliance)

gbm_tuned_1 = GradientBoostingClassifier(learning_rate = 0.0125,
                                         n_estimators=320,
                                         min_samples_leaf=30,
                                         min_samples_split=300,
                                         max_depth=9,
                                         max_features=18,
                                         subsample=0.85,
                                         random_state=5100)

cross_val_score(gbm_tuned_1,train,train_merged.compliance, scoring='roc_auc')

#lr.fit(train,train_merged.compliance)
#y_pred_proba = lr.predict_proba(test)

#del train,test,train_merged,auc_cross,categorical_variables,numeric_variables,address_merged

#answer = pd.Series(y_pred_proba[:,1], index = test_merged.ticket_id)

#    return answer


# In[ ]:

blight_model()