
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas import DatetimeIndex
from sklearn.model_selection import cross_val_score


# In[16]:


df = pd.read_csv('../../data/merged-train-data.csv')
df[0::100000]


# In[17]:


#creating the feature matrix
feature_cols = ['Store', 'Temperature','Fuel_Price','CPI','Unemployment', 'Markdowns', 'Size', 'Dept', 'Weekly_Sales', 'Year', 'Month', 'Day']
X = df[feature_cols]
y = df['IsHoliday']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=2)


# In[18]:


logreg = LogisticRegression()


# In[19]:


logreg.fit(X_train, y_train)


# In[20]:


y_predict_logreg = logreg.predict(X_test)
print(y_predict_logreg)


# In[21]:


score_logreg = accuracy_score(y_test, y_predict_logreg)
print(score_logreg)


# In[22]:


accuracy_list = cross_val_score(logreg, X, y, cv=5, scoring='accuracy')


# In[23]:


accuracy_cv = accuracy_list.mean()


# In[24]:


print(accuracy_cv)
