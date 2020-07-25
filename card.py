# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:49:40 2020

@author: anuj8
"""
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# for performing statistical tests and comparing different models
import statsmodels.formula.api as sm
from scipy import stats
# for saving matplotlib plots as pdf
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
# ????
from patsy import dmatrices


#% matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# to print arrays upto its max size we write the following line of code.
np.set_printoptions(threshold = sys.maxsize)
np.set_printoptions(threshold = 3)
# to display complete dataset
pd.options.display.max_columns = None
pd.options.display.max_rows = None

sns.set(style = 'whitegrid')


# plot settings
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

data = pd.read_excel('Dataset.xlsx')
data.head()

data.columns

#lets create the dependent variable

data['total_spent'] = data['cardspent'] + data['card2spent']
data['total_spent']


# Removing features which are irrelevant

new = data.drop(['custid','cardspent','card2spent'],axis = 1)

for x in ['region','townsize','gender','agecat','edcat','birthmonth','jobcat','union','employ','empcat','retire',
          'inccat','default','jobsat','marital','spousedcat','homeown','hometype','address','addresscat','cars','carown',
          'cartype','carcatvalue','carbought','carbuy','commute','commutecat','commutecar',
          'commutemotorcycle','commutecarpool','commutebus','commuterail','commutepublic','commutebike','commutewalk',
         'commutenonmotor','telecommute','reason','polview','polparty','polcontrib','vote','card','cardtype','cardbenefit',
         'cardfee','cardtenure','cardtenurecat','card2','card2type','card2benefit','card2fee','card2tenure','card2tenurecat',
         'active','bfast','churn','tollfree','equip','callcard','wireless','multline','voice','pager','internet','callid',
         'callwait','forward','confer','ebill','owntv','ownvcr','owndvd','owncd','ownpda','ownpc','ownipod','owngame','ownfax',
        'news','response_01','response_02','response_03']:
                      new[x]=new[x].astype('object')


cat = ['region','townsize','gender','agecat','edcat','birthmonth','jobcat','union','employ','empcat','retire',
          'inccat','default','jobsat','marital','spousedcat','homeown','hometype','address','addresscat','cars','carown',
          'cartype','carcatvalue','carbought','carbuy','commute','commutecat','commutecar',
          'commutemotorcycle','commutecarpool','commutebus','commuterail','commutepublic','commutebike','commutewalk',
         'commutenonmotor','telecommute','reason','polview','polparty','polcontrib','vote','card','cardtype','cardbenefit',
         'cardfee','cardtenure','cardtenurecat','card2','card2type','card2benefit','card2fee','card2tenure','card2tenurecat',
         'active','bfast','churn','tollfree','equip','callcard','wireless','multline','voice','pager','internet','callid',
         'callwait','forward','confer','ebill','owntv','ownvcr','owndvd','owncd','ownpda','ownpc','ownipod','owngame','ownfax',
        'news','response_01','response_02','response_03']



num = new.select_dtypes(exclude=['object']).columns

new_num = new[num]


new_cat = new[cat] 


#### CREATING A DATA AUDIT REPORT ####

def var_summary(x):
    return pd.Series([x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.dropna().quantile(0.01), x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75), x.dropna().quantile(0.90),x.dropna().quantile(0.95), x.dropna().quantile(0.99),x.max()], 
                     index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX'])


num_summary = new_num.apply(lambda x : var_summary(x), axis = 0)


def var1_summary(x):
    return pd.Series([x.count(),x.isnull().sum(),x.value_counts(), x.unique()],
                      index = ['N', 'NMISS' , 'Frequency of values', 'Unique values'])
    
cat_summary = new_cat.apply(lambda x : var1_summary(x), axis = 0).T


# Now we need to handle outliers

def outlier_capping(x):
    x = x.clip(x.quantile(0.99),x.quantile(0.01))
    return x


new_num1 = new_num.apply(lambda x: outlier_capping(x))

# Handling missing values
# Now since we have removed extreme values(i.e. the outliers, we can replace missing values with mean)


def impute(x):
    x = x.fillna(x.mean())
    return x

print("count of missing values : ", new_num1.isnull().sum().values.sum())


new_num1 = new_num1.apply(lambda x : impute(x))

num_summary_1 = new_num1.apply(lambda x : var_summary(x),axis = 0)


# Handling Missing values for categorical variables


def impute_1(x):
    x = x.fillna(x.mode().iloc[0])
    return x


new_cat1 = new_cat.apply(lambda x :impute_1(x))

cat_summary_1 = new_cat1.apply(lambda x : var1_summary(x)).T


# checking for missing values


print("The total number of missing values in the numerical dataframe is", new_num1.isnull().sum().values.sum())
print("The total number of missing values in the categorical dataframe is", new_cat1.isnull().sum().values.sum())


dummy = pd.get_dummies(new_cat1['region'])

# Now lets create dummy variables for all the categorical variables.

def create_dummies(df,colname):
    col_dummies = pd.get_dummies(df[colname],prefix = colname)
    col_dummies.drop(col_dummies.columns[0],axis = 1,inplace = True)
    df = pd.concat([df,col_dummies], axis = 1)
    df.drop(colname, axis = 1, inplace = True)
    return df

custdata_df_cat=new.select_dtypes(include=['object'])

cat_varlist=list(custdata_df_cat.columns)

for c_feature in cat_varlist:
    custdata_df_cat[c_feature]=custdata_df_cat[c_feature].astype('category')
    custdata_df_cat=create_dummies(custdata_df_cat,c_feature)

#lets concatenate the numerical features and categorical features

final = pd.concat([new_num1,custdata_df_cat],axis = 1) 
    
# Now lets check whether our dependent variable is normally distributed or not.

import pylab

stats.probplot(final['total_spent'],dist = 'norm',plot = pylab)

# Since the points are not concentrated along the 45 degree line, this means that the variable is not normally distributed.
  
sns.distplot(final['total_spent'])     

# we will apply boxcocx transform on the dependent variable to make it normally distributed.

final['box_cox_total_spent'],fitted_lambda = stats.boxcox(final['total_spent'])
sns.distplot(final['box_cox_total_spent'])

# we now have a normally distributed dependent variable. Lets drop total_spent
final = final.drop('total_spent',axis = 1)
z = final.drop('box_cox_total_spent',axis = 1)



from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test = train_test_split(a,final['box_cox_total_spent'],test_size = 0.3, random_state = 12)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Now we will be Random forest regressor to get the best set of features
# To determines the number of trees in the Random Forest, we will be using hyperparamter tuning


no_of_trees = {'n_estimators':np.arange(10,25)} # 'n_estimators is a key word'

tree = GridSearchCV(RandomForestRegressor(oob_score = False,warm_start = True),no_of_trees,cv = 2)

#tree = GridSearchCV(RandomForestRegressor(oob_score=False,warm_start=True),param_grid,cv=2)

tree.fit(x_train,y_train)


tree.best_params_ 
# Indicates the number of trees.

r = RandomForestRegressor(oob_score = True,n_estimators = 23)

r.fit(x_train,y_train)

r.oob_score_

type(r.feature_importances_)
# it is an array

# getting the indices of the features in the descending(most important first) order
indices = np.argsort(r.feature_importances_)[::-1]

table = pd.DataFrame(columns = ['Rank','Feature_name','Importance'])

for x in range(x_train.shape[1]):
    table.loc[x] = x+1,x_train.columns[indices[x]],r.feature_importances_[indices[x]]
    
# Selecting the first 75 features    
##################################################################
rf_features = list(table['Feature_name'].loc[0:75])
rf_features_1 = list(table['Feature_name'].loc[0:75])
##################################################################

# Adding the dependent variable in the list of features


rf_features.append('box_cox_total_spent')

a = final[rf_features]
b = final[rf_features_1]

# Lets check for multicollinearity amon the features

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X = add_constant(b) # to add a column of 1's 

scores = pd.DataFrame(pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns))

scores['index']

# Dropping the variables from the series having VIF score > 10
new_scores = scores[scores[0]<10.0]

feature = new_scores.index.values

data = final[feature]

data = pd.concat([data,final['box_cox_total_spent']],axis = 1)


# we will use OLS regression technique.
x_train,x_test,y_train,y_test = train_test_split(data.drop(['box_cox_total_spent'],axis = 1),data['box_cox_total_spent']
                     ,test_size = 0.2, random_state = 123)

# x_train = sm.add_constant(x_train)
# y_train = sm.add_constant(y_train)

lm=sm.OLS(y_train,x_train).fit()

lm.summary()

# testing the model on the test data.

y_pred = lm.predict(x_test)
x_pred = lm.predict(x_train)

print("MSE TEST",metrics.mean_squared_error(y_pred,y_test))
print("MSE TRAIN",metrics.mean_squared_error(x_pred,y_train))

residuals = y_pred - y_test


sns.distplot(residuals)
# The residuals are normally distributed indicating that the model is working properly with the test data.


# Now we will run the model for the entire dataset pre spliting for the linear regression model.
# Variable data contains the required data
from scipy.special import inv_boxcox

data1 = data.drop(['box_cox_total_spent'],axis = 1)

credit_card_sales = inv_boxcox(lm.predict(data1),fitted_lambda)

final_output = pd.concat([data1,credit_card_sales],axis = 1)

# exporting the final file as csv
final_output.to_csv('out.csv',index = False)
