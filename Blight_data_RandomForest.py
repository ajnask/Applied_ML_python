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

## Logistic Regression Implementation
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

from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score

rf = RandomForestClassifier(random_state =5100)
auc_cross = cross_val_score(rf,train,train_merged.compliance,cv =5,scoring = 'roc_auc') 
#param = {'n_estimators':[50,100,200], 'max_features' : [2,4,8,10,14,16]}
#lr_gridsearch = GridSearchCV(rf,param_grid=param, cv =3, scoring = 'roc_auc')
#lr_gridsearch.fit(train,train_merged.compliance)

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(train,train_merged.compliance,random_state = 5100)
#rf.fit(X_train,y_train)
#y_pred = rf.predict_proba(X_test)

#roc_score = roc_auc_score(y_test,y_pred[:,1])
#lr.fit(train,train_merged.compliance)
#y_pred_proba = lr.predict_proba(test)

#0.791780065 8 50
#del train,test,train_merged,auc_cross,categorical_variables,numeric_variables,address_merged

#answer = pd.Series(y_pred_proba[:,1], index = test_merged.ticket_id)
#answer = (lr_gridsearch.best_params_,lr_gridsearch.best_score_)

#    return answer



#bm = blight_model()
#res = 'Data type Test: '
#res += ['Failed: type(bm) should Series\n','Passed\n'][type(bm)==pd.Series]
#res += 'Data shape Test: '
#res += ['Failed: len(bm) should be 61001\n','Passed\n'][len(bm)==61001]
#res += 'Data Values Test: '
#res += ['Failed: all values should be in [0.,1.]\n','Passed\n'][all((bm<=1.) & (bm>=0.))]
#res += 'Data Values type Test: '
#res += ['Failed: bm.dtype should be float\n','Passed\n'][str(bm.dtype).count('float')>0]
#res += 'Index type Test: '
#res += ['Failed: type(bm.index) should be Int64Index\n','Passed\n'][type(bm.index)==pd.Int64Index]
#res += 'Index values type Test: '
#res += ['Failed: type(bm.index[0]) should be int64\n','Passed\n'][str(type(bm.index[0])).count("int64")>0]
#
#res += 'Output index shape test:'
#res += ['Failed, bm.index.shape should be (61001,)\n','Passed\n'][bm.index.shape==(61001,)]
#
#res += 'Output index test: '
#if bm.index.shape==(61001,):
#    res +=['Failed\n','Passed\n'][all(pd.read_csv('test.csv',usecols=[0],index_col=0).sort_index().index.values==bm.sort_index().index.values)]
#else:
#    res+='Failed'
#print(res)



# In[ ]:

blight_model()


# Cluster analysis on space
from sklearn.cluster import KMeans

#Visualizing lat - lon coordinates
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(latlon.lat,latlon.lon)
plt.xlabel('Latitude')
plt.ylabel('Longitude')


# Number of clusters
scores = []
k_range = np.linspace(1,20, num =20 , dtype = int)
for k in k_range:
    kmeans = KMeans(n_clusters=k).fit(X.loc[:,['lat','lon']])
    scores.append(kmeans.score(X.loc[:,['lat','lon']]))

plt.figure()
plt.plot(range(50000),scores)


k = 4
kmeans = KMeans(n_clusters = k)
X = latlon[pd.notnull(latlon.lat)]
#Fitting the data
kmeans.fit(X.loc[:,['lat','lon']])
# Cluster centers
x, y = list(zip(*kmeans.cluster_centers_))
#kmeans visualization
X['cluster'] = kmeans.labels_
plt.scatter(X.lat[X.cluster == 0],X.lon[X.cluster == 0],color = 'green')
plt.scatter(X.lat[X.cluster == 1],X.lon[X.cluster == 1],color = 'blue')
plt.scatter(X.lat[X.cluster == 2],X.lon[X.cluster == 2],color = 'grey')
plt.scatter(X.lat[X.cluster == 3],X.lon[X.cluster == 3],color = 'yellow')
plt.scatter(x,y, color = 'red', s = 40) 
