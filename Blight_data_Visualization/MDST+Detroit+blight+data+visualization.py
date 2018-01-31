
# coding: utf-8

# ## Understanding Property Maintenance Fines from MDST Blight Data
# 
# This assignment is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)).That was also a part of Applied Data Science with Python Specialization offered by the ([University of Michigan](https://www.umich.edu/)) via ([Coursera](https://www.coursera.org/learn/python-machine-learning/home/welcome)). 
# 
# The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?
# 
# All data for this assignment has been provided through the [Detroit Open Data Portal](https://data.detroitmi.gov/).
# ___
# 
# <br>
# 
# **File descriptions**
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
# ## Visualization
# 
# This part of the project **only deals with Exploratory Data Analysis**. The predictive modeling part will be dealt with seperately. For now, I'll look at the data and try to understand it.
# 
# Also, I'm not looking into the geolocational data right now. Though it would be a strong feature in predictive modeling. But as far as thing part of the assignment is concerned, I'm only looking at the training data. 
# 
# ___
# 
# ## Loading Data sets
# 
# Let's load the datasets and take a peek at the data given

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')

# Loading Datasets
address = pd.read_csv('addresses.csv')
train = pd.read_csv('train.csv', encoding = 'ISO-8859-1', low_memory = False)
latlon = pd.read_csv('latlons.csv')
display(train.head(5))
display(latlon.head(5))
display(address.head(5))


# It's no use keeping these datasets seperate. Merging them into a single dataset would make the analysis a lot easier

# In[10]:


# Merging address and latlon datasets
address_merged = pd.merge(address,latlon, how = 'inner',on = 'address')

# Merging train with address_merged 
train_merged = pd.merge(train,address_merged,how = 'inner', on = 'ticket_id')
display(train_merged.head(5))

# Deleting the previous datasets to save memory
del address,train,latlon


# The compliance table has three variables as described in the file description. But since we are looking into binary classification of compliance and non-compliance, we can ignore the null values (i.e. not responsible). And then let's look at the data overall.

# In[11]:


train_merged = train_merged[pd.notnull(train_merged.compliance)]

# Overall
train_merged.info()
train_merged.describe()


# Since violation_zip_code,grafitti_status and non_us_str_code are almost empty.  
# Let's ignore those variables too. Now let's look at the num.  
# ## Single Variable Visualization
# 
# ### Numerical Variables

# In[12]:


# Fine amount
plt.figure()
plt.hist(train_merged.fine_amount, bins = 50)
plt.title('Fine Amount')

# Admin Fee CONSTANT
plt.figure()
plt.hist(train_merged.admin_fee,bins = 100)
plt.title('Admin Fee')

# State Fee CONSTANT
plt.figure()
plt.hist(train_merged.state_fee,bins = 100)
plt.title('State Fee')

# Late Fee
plt.figure()
plt.hist(train_merged.late_fee,bins = 100)
plt.title('Late Fee')

# Discount Amount CONSTANT
plt.figure()
plt.hist(train_merged.discount_amount,bins = 100)
plt.title('Discount Amount')

# Clean up cost CONSTANT
plt.figure()
plt.hist(train_merged.clean_up_cost,bins = 100)
plt.title('Clean up cost')

#This turns out to be zero for all the cases

# Judgment amount
plt.figure()
plt.hist(train_merged.judgment_amount,bins = 100)
plt.title('Judgment Amount')

# Payment amount
plt.figure()
plt.hist(train_merged.payment_amount,bins = 100)
plt.title('Payment Amount')

# Balance amount
plt.figure()
plt.hist(train_merged.balance_due,bins=100)
plt.title('Balance Due')


# That wasn't particularly informative. Althought we did realise that the Admin Fee, State Fee, Discount Amount and Clean up cost are all constants where Discount Amount and Clean up cost being zero.  
#   
# ### Agency names

# In[14]:


# Agency name
agency = train_merged.agency_name.groupby(train_merged.agency_name).size()
plt.figure(figsize = (9,5))
plt.bar(agency.index,agency, width = 0.4)
plt.title('Agency Names')
xticks_agency = ['Buildings,\nSafety Engineering\n& Env Department','Department\nof Public Works',
          'Detroit\nPolice Department','Health Department','Neighborhood\nCity Halls']
plt.gca().set_xticklabels(xticks_agency);


# 
# 
# Alright, that's informative.   
# ** We have a lot more violations in Buildings,Safety Engineering and Environmental Department than every other department**
# 
# ### Inspector Names
# Now let's look at the Names of the inspectors who dealt the violations and see the top ten list

# In[15]:


# inspector_name
inspector = train_merged.inspector_name.groupby(train_merged.inspector_name).size().sort_values(ascending = False)
plt.figure(figsize = (8,5))
inspector[:10].plot(kind = 'bar')
plt.title('Inspectors')
ax = plt.gca()
for item in ax.get_xticklabels():
    item.set_rotation(90)
plt.subplots_adjust(bottom = 0.3)


# Most number of the violations seems to be filed by Mr. Morris John. This isn't a worth while information, unless we have the background data on these inspectors. For now, let this be just fun information on who filed the most violation cases.
# 
# ### Country
# 
# The country variable contains the country given in the mailing address. If we consider the mailing address as a key to figure out whether the violator was an immigrant, there seems to be some who are. Although we cannot infer any conclusions from the dataset since the number of non-us violators are comparitively rare.

# In[16]:


# country
country = train_merged.country.groupby(train_merged.country).size().sort_values(ascending = False)
print(country)


# ### Violation Code

# In[17]:


# violation_code
viol_code = train_merged.violation_code.groupby(train_merged.violation_code).size().sort_values(ascending = False)
plt.figure()
viol_code[:10].plot(kind = 'bar')
plt.subplots_adjust(bottom = 0.3)
plt.title('Violation Code')

description = []
for i in viol_code[:10].index:
    temp = train_merged[train_merged.violation_code == i]['violation_description'].unique()
    description.append([i,str(temp)])
description = pd.DataFrame(description, columns = ['Violation code','Description'])

pd.options.display.max_rows
pd.set_option('display.max_colwidth', -1)
display(description)


# It look like the most number of violations are done for obtaining **certificate of compliance** and **certificate for registration of rental property**.
# 
# ### Disposition
# 
# Looking at the judgment types, we can infer that the default judgment is Responsible by default.

# In[69]:


# disposition
dispo = train_merged.disposition.groupby(train_merged.disposition).size().sort_values(ascending = False)
plt.figure()
dispo.plot(kind = 'bar')
xticks_dispo = ['Responsible\nby\nDefault','Responsible\nby\nAdmission',
          'Responsible\nby\nDetermination','Responsible\n(Fine Waived)\nby\nDetermination']
plt.gca().set_xticklabels(xticks_dispo, rotation = 0)
plt.title('Disposition');


# ## Variable Interaction
# 
# Let's look at how the variables interact with each other, i.e. how does each variable correlate.
# 
# ### Numerical Variables
# Let's look at a correlation matrix of all the payments and see if there is any correlation between payments

# In[113]:


# Correlogram
numeric_variables = ['balance_due','judgment_amount','late_fee','payment_amount','fine_amount']
pd.plotting.scatter_matrix(train_merged.loc[:,numeric_variables],figsize = (9,9));
train_merged.loc[:,numeric_variables].corr()


# The figure shows that some of the payment amounts are correlated to each other. For eg, Judgment amount and fine amount seems to be positively correlated and almost equal. This can also be seens from the correlation table.
# 
# ### Compliance vs Payments
# 
# Now let's look at how these payments correlate with compliance 

# In[116]:


#numeric_variables = ['balance_due','judgment_amount','late_fee','payment_amount','fine_amount']
# Boxplots
# Compliance Vs Payment Amounts

for item in numeric_variables:
    train_merged.boxplot(item,by = 'compliance')
    #plt.yscale('log')


# These plots are not particulary informative because of the scaling used. But for now, let's leave it at that.
# 
# ### Compliance vs Street
# Let's look at the number and percentage of compliance per street. Here we are looking at the top 10 streets with most number of non-compliance, compliance and percentage compliance respectively.

# In[134]:


# Let's look at the number of non-compliance per street
per_street = pd.crosstab(train_merged.violation_street_name,train_merged.compliance)
per_street_non_compliance = per_street[0].sort_values(ascending = False)

plt.figure(figsize = (9,6))
per_street_non_compliance[:10].plot(kind = 'bar', color = 'red')
plt.title('Non - compliance (Top 10)')
plt.subplots_adjust(bottom = 0.25)

per_street_compliance = per_street[1].sort_values(ascending = False)
plt.figure(figsize = (9,6))
per_street_compliance[:10].plot(kind = 'bar')
plt.title('Compliance (Top 10)')
plt.subplots_adjust(bottom = 0.25)

# Percentage of compliance per street
per_street_compliance_percent = (per_street[1]/per_street.sum(axis =1))*100
per_street_compliance_percent = per_street_compliance_percent.sort_values(ascending = False)
plt.figure(figsize = (9,6))
per_street_compliance_percent[:30].plot(kind = 'bar')
plt.title('Percentage Compliance')
plt.subplots_adjust(bottom = 0.25)


# The most number of compliance and non-compliance are from the same street, **Seven Miles**. But it doesn't have a considerable compliance percentage. There are a lot of streets with 100% compliance. Although, some of them are 100% compliant just because the number of violations are very low.

# #### Note to Data Imbalance:
# From the table below, it is clear that we hace a lot more observations of non-compliance than compliance. As a result, from this points onwards, it is better to look at just compliance and ignore non-compliance, just to get an understanding of the data set and where and what correlates with compliance.

# In[8]:


train_merged.compliance.groupby(train_merged.compliance).size()


# ### Compliance vs Agency

# In[44]:


# agency_name
per_agency = pd.crosstab(train_merged.agency_name,train_merged.compliance)
per_agency_compliance = (per_agency[1]/per_agency.sum(axis =1))*100
plt.figure()
per_agency_compliance.plot(kind = 'bar', figsize = (9,6))
plt.gca().set_xticklabels(xticks_agency, rotation = 0)
plt.title('Percentage Compliance per agency')
plt.subplots_adjust(bottom = 0.2)


# Although most violation cases turns out to be non-compliant, more than 12% of **Detroit Police Department** cases are compliant. We initially saw that most of the violation cases are form Buildings,Safety Engineering and Env department, but only a mere 6% of it is compliant.
# 
# ### Compliance vs Inspector

# In[58]:


# inspector_name
per_inspector_name = pd.crosstab(train_merged.inspector_name, train_merged.compliance)
per_inspector_compliance = (per_inspector_name[1]/per_inspector_name.sum(axis =1))*100
per_inspector_compliance = per_inspector_compliance.sort_values(ascending = False)
plt.figure()
per_inspector_compliance[:10].plot(kind = 'bar', figsize = (9,6))
plt.subplots_adjust(bottom = 0.3)


# ### Compliance vs Violation type

# In[59]:


# violation_code
per_violation_code = pd.crosstab(train_merged.violation_code, train_merged.compliance)
per_violation_compliance = (per_violation_code[1]/per_violation_code.sum(axis =1))*100
per_violation_compliance = per_violation_compliance.sort_values(ascending = False)
plt.figure()
per_violation_compliance[:10].plot(kind = 'bar',figsize = (9,6))
plt.title('Percentage Compliance per Violation type')
plt.subplots_adjust(bottom = 0.3)

compliance_description = []
for i in per_violation_compliance[:10].index:
    temp = train_merged[train_merged.violation_code == i]['violation_description'].unique()
    compliance_description.append([i,str(temp)])

compliance_description = pd.DataFrame(compliance_description, columns = ['Violation code','Description'])

pd.options.display.max_rows
pd.set_option('display.max_colwidth', -1)
display(compliance_description)


# It looks like most people okay with paying for insufficient parking violations, almost 100% of parking ticket violations are compliant. Although the number of these violations may play a big hand in predictive modeling. But every other violations such as unlawful disposal of commercial solid waste, clearance for rental property etc. are only 40% compliant.
# 
# ### Compliance vs Disposition

# In[68]:


# disposition
per_dispo = pd.crosstab(train_merged.disposition, train_merged.compliance)
per_dispo_compliance = ((per_dispo[1]/per_dispo.sum(axis = 1))*100).sort_values(ascending = False)
plt.figure()
per_dispo_compliance.plot(kind = 'bar')
plt.gca().set_xticklabels(xticks_dispo[::-1], rotation = 0)
plt.subplots_adjust(bottom = 0.3)


# ## Inferences:
# - **Since the number of compliant observations are very low in the data set, it looks as if almost all the violations turned out to be non-compliant. So it becomes important to consider metrics other than accuracy while doing predictive modelling**
# - The Admin Fee, State Fee, Discount Amount and Clean up cost are all constants and Discount Amount and Clean up cost are zero.
# - There are lot more violations in Buildings,Safety Engineering and Environmental Department than every other department
# - Most number of violations are done for obtaining certificate of compliance and certificate for registration of rental property.
# - Some of the payment amounts are correlated to each other. For eg, Judgment amount and fine amount seems to be positively correlated and are almost equal
# - The most number of compliance and non-compliance are from the same street, Seven Miles. But it doesn't have a considerable compliance percentage. There are a lot of streets with 100% compliance
# - More than 12% of Detroit Police Department cases are compliant. We initially saw that most of the violation cases are form Buildings,Safety Engineering and Env department, but only a mere 6% of it is compliant.
# - It looks like most people okay with paying for insufficient parking violations, almost 100% of parking ticket violations are compliant (This could also be because of the small number of violations in that particular type). 
# - Every other violations such as unlawful disposal of commercial solid waste, clearance for rental property etc. are only 40% compliant.
# - It is to be noted that the percentage of compliance for responsible by default is the least among all four judgments.
# 
