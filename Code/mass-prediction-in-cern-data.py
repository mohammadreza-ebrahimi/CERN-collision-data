#!/usr/bin/env python
# coding: utf-8

# * # **Electron Mass Prediction**

# The essential liberaries

# In[77]:


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib as mlp
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler # OneHE is used for convert 'str' data to numerical
#from sklearn.impute import SimpleImputer
#from sklearn.compose import ColumnTransformer
#from sklearn.pipeline import Pipeline  # multi preproseccing
#from sklearn.base import BaseEstimator, TransformerMixin  #use to construct eg. AttributesAdder 
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# In[78]:


Data = pd.read_csv("/home/mohammadreza/Downloads/dielectron.csv")
Data.head(10)


# Getting some useful information about the data

# In[49]:


Data.info()


# Seems it containes 85 NaN or NA value in the label data *Mass*

# In[50]:


Data.describe()


# In[51]:


Data.shape


# Plotting the Histogram can be so useful in order to construct best training and test set.

# In[11]:


Data.hist(bins=50, figsize=(20, 15))
plt.show()


# Looks scaling data is needed

# In[4]:


corr_matrix = Data.corr()
print(corr_matrix['M'].sort_values(ascending=False))


# **Adding attribute**

# In[52]:


Data['E_total'] = Data['E1'] + Data['E2']
# Data['pt2_per_E2']=Data['pt2']/Data['E2']


# Drop some useless features

# In[79]:


Data.drop('Event', axis=1, inplace=True)
Data.drop('Run', axis=1, inplace=True)


# In[80]:


corr_matrix = Data.corr()
print(corr_matrix['M'].sort_values(ascending=False))


# In[81]:


Data.plot(kind='scatter', x='M', y='pt2', alpha=0.15)
plt.show()


# In[7]:


plt.hist(Data['M'], bins=50)
plt.show()


# Visualizing data

# In[23]:


plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='YlGnBu')
sns.
plt.title('Correlation Matrix')
plt.show()


# In[8]:


plt.hist(Data['pt1'], bins=100)
plt.show()


# In[82]:


Data['pt1_cat'] = pd.cut(Data['pt1'],
                         bins=[0, 10, 20, 30, 40, 50, np.inf],
                         labels=[1, 2, 3, 4, 5, 6])
plt.hist(Data['pt1_cat'])
plt.show()


# Splitting *Training set* and *Test set* based on categorized feature

# In[83]:



st_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in st_split.split(Data, Data['pt1_cat']):
    train_set = Data.loc[train_index]
    test_set = Data.loc[test_index]
    
train_set


# In[84]:


for set_ in (train_set, test_set):
    set_.drop('pt1_cat', axis=1, inplace=True)


# In[85]:


train_set.info()


# In[86]:


train_set2 = train_set.dropna(subset=['M'], inplace=False)
train_set2.info()


# In[17]:


from pandas.plotting import scatter_matrix

attributes = ['pt1', 'pt2', 'E1', 'E2', 'E_total', 'M']
scatter_matrix(train_set[attributes], figsize=(20, 15))
plt.show()


# Now, we have to separate the *label data* and the *predictors*

# In[87]:


train_features = train_set2.drop('M', axis=1, inplace=False)
train_label = train_set2['M'].copy()


# Scaling the data is caused to have more precise model training

# In[88]:


scale = StandardScaler()
cern_prepared = scale.fit_transform(train_features)


# > **PolynomialFeatures** are those features created by raising existing features to an exponent. The **degree** of the polynomial is used to control the number of features added, e.g. a degree of 3 will add two new variables for each input variable. Typically a small degree is used such as 2 or 3.

# In[17]:


#from sklearn.preprocessing import PolynomialFeatures

#cern_poly = PolynomialFeatures(2)
#cern_prepared_2 = cern_poly.fit_transform(cern_prepared)


# In[89]:


from sklearn.preprocessing import PolynomialFeatures
cern_poly = PolynomialFeatures(2)
data_perpared = cern_poly.fit_transform(train_features)


# We could use `Pipeline` in order to prepare our training data with just one command which has been included both `SatandardScalar` and `PolynomialFeatures`. Since we want to show various models, here I did not use that. 

# * ****Training Model**** \
# As the first Model, We try the **LinearRegresion** model for linear equation (cern_prepared)

# In[90]:


lin_reg = LinearRegression()
lin_reg.fit(cern_prepared, train_label)


# In[91]:


scores = cross_val_score(lin_reg, cern_prepared, train_label,
                         scoring='neg_mean_squared_error',
                         cv=20)
lin_rmse = np.sqrt(-scores)
print(lin_rmse)


# In[92]:


def display_score(scores):
    print('Scores:', scores),
    print('Mean:', scores.mean()),
    print('Std:', scores.std())


display_score(lin_rmse)


# In[106]:


from sklearn.linear_model import Ridge

ridge_reg=Ridge(alpha=1, solver='cholesky')
ridge_reg.fit(cern_prepared, train_label)


# In[107]:


scores = cross_val_score(ridge_reg, cern_prepared, train_label,
                         scoring='neg_mean_squared_error',
                         cv=10)
lin_rmse = np.sqrt(-scores)
display_score(lin_rmse)


# In[116]:


from sklearn.linear_model import Lasso

lasso_reg=Lasso(alpha=0.01)
lasso_reg.fit(cern_prepared, train_label)


# In[117]:


scores = cross_val_score(lasso_reg, cern_prepared, train_label,
                         scoring='neg_mean_squared_error',
                         cv=10)
lin_rmse = np.sqrt(-scores)
display_score(lin_rmse)


# It seemsthat we have underfitting, that is not good model. Let's try another one. 

#  What about Tree?\
# **DecisionTreeRegressor** as the second model

# In[21]:


tree_reg = DecisionTreeRegressor()
tree_reg.fit(cern_prepared, train_label)


# In[22]:


tree_scores = cross_val_score(tree_reg, cern_prepared, train_label,
                              cv=10, scoring="neg_mean_squared_error")
tree_rmse = np.sqrt(-tree_scores)
print(tree_rmse)


# In[23]:


display_score(tree_rmse)


# Tha is not bad but it seems we still have underfitting\
# But let's try third one, **RandomForestRegressor**

# In[25]:


forest_reg = RandomForestRegressor()
forest_reg.fit(cern_prepared, train_label)


# In[26]:


forest_score = cross_val_score(forest_reg, cern_prepared, train_label,
                               cv=5, scoring='neg_mean_squared_error')
forest_rmse = np.sqrt(-forest_score)
print(forest_rmse)


# In[27]:


display_score(forest_rmse)


# * **PolynomialFeatures**

# It could be a good result, but let us examine these models with our *cern_prepared_2* which has been prepared by **PolynomialFeatures**.\
# First for **LinearRegression**

# In[15]:


lin_reg = LinearRegression()
lin_reg.fit(data_perpared, train_label)


# In[16]:


poly_scores = cross_val_score(lin_reg, data_perpared, train_label,
                              scoring="neg_mean_squared_error", cv=10)
poly_scores_rmse = np.sqrt(-poly_scores)
print(poly_scores_rmse)


# In[18]:


def display_score(scores):
    print('Scores:', scores),
    print('Mean:', scores.mean()),
    print('Std:', scores.std())


display_score(poly_scores_rmse)


# Wow! It seems that we have second order model.\
# Let's do it for **DecisionTreeRegressor**

# In[19]:


tree_reg = DecisionTreeRegressor()
tree_reg.fit(data_perpared, train_label)


# In[27]:


tree_poly_scores = cross_val_score(tree_reg, data_perpared, train_label,
                        scoring='neg_mean_squared_error', cv=10)
tree_poly_rmse = np.sqrt(-tree_poly_scores)
print(tree_poly_rmse)


# In[28]:


display_score(tree_poly_rmse)


# *Looks very good*, It could be used for our main model due to testing with cross validation,
# Let's try **RandomForestRegressor**

# In[40]:


forest_reg = RandomForestRegressor()
forest_reg.fit(data_perpared, train_label)


# In[43]:


forest_prediction = forest_reg.predict(data_perpared)
forest_mes=mean_squared_error(forest_prediction, train_label)
forest_rmse=np.sqrt(forest_mes)
forest_rmse


# _Do we have Overfitting ?_ Lest's test it with `Cross_val_score`

# In[30]:


forest_poly_scores = cross_val_score(forest_reg, data_perpared, train_label,
                        scoring='neg_mean_squared_error', cv=5)
forest_poly_rmse = np.sqrt(-forest_poly_scores)
print(forest_poly_rmse)


# In[31]:


display_score(forest_poly_rmse)


# It is perfect result. We can use it as the best model and now we proceed in **Tunung Hyperparameters**.\
# Before it, let's try it for the **test_set**

# In[44]:


test_data = test_set.dropna(subset=['M'])
test_X = test_data.drop("M", axis=1)
test_y = test_data["M"].copy()
test_X_poly = cern_poly.transform(test_X)


# In[46]:


final_predictions = forest_reg.predict(test_X_poly)
#final_predictions = tree_reg.predict(test_X_poly)
final_mse = mean_squared_error(final_predictions, test_y)
final_rmse = np.sqrt(final_mse)
final_rmse


# *For getting a more precise regression model, we can tune hyperparameter as our next step...*\
# We would make it as soon as possible.
