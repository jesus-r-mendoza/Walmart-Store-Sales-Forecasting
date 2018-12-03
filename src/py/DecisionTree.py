
# coding: utf-8

# # Decision Tree

# ### Merge feature and test datasets

# In[1]:


import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from pandas import DatetimeIndex
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('../data/merged-train-data.csv')
df[0::100000]


# In[3]:


#creating the feature matrix 
def categorical_to_numeric(x):
    if x == False:
        return 0
    elif x == True:
        return 1
    
df['IsHoliday'] = df['IsHoliday'].apply(categorical_to_numeric)

y =df['Weekly_Sales']
X = df.drop(columns=['Weekly_Sales','Type'])


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)


# ## Decision Tree

# In[5]:


#Decision Tree
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)


# ## Testing on the testing set

# In[6]:


#Our predictions
y_predict_dt = regressor.predict(X_test)
print(y_predict_dt)


# ## Accuracy Evaluation

# In[7]:


#Accuracy score based on predictions
#score_dt = accuracy_score(y_test, y_predict_dt)

score_dt = regressor.score(X_test, y_test)
print(score_dt)


# ## Feature Importance

# In[10]:


for name, importance in zip(X.columns, regressor.feature_importances_):
    print(name, importance)

