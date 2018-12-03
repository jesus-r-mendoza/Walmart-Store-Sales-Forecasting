
# coding: utf-8

# ### Linear Regression with Coefficient and RMSE

# ## Merge feature and test datasets

# In[1]:


### Importing the required packages and libraries
# we will need numpy and pandas later
import numpy as np
import pandas as pd


from sklearn.linear_model import LinearRegression
#import DecisionTreeClassifier class
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Randomly splitting the original dataset into training set and testing set
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


# In[2]:


# reading a CSV file directly from Web, and store it in a pandas DataFrame:
# "read_csv" is a pandas function to read csv files from web or local device:
# walmart_features_df =  pd.read_csv('features.csv')

# walmart_sampleSubmission_df =  pd.read_csv('sampleSubmission.csv')

# walmart_stores_df =  pd.read_csv('stores.csv')

# walmart_test_df =  pd.read_csv('test.csv')

# walmart_train_df =  pd.read_csv('train.csv')


# In[3]:


df = pd.read_csv('../../data/merged-train-data.csv')
df[0::100000]


# ### Split the merged data set into X_train and y_train, use test dataset for X_test

# In[4]:


#split with method

y =df['Weekly_Sales']
X = df.drop(columns=['Weekly_Sales','Type'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=2)


# In[5]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(X_train.head())


# In[6]:


#Linear Regression

linear_reg = LinearRegression()
linear_reg.fit(X_train,y_train)


# In[7]:


print("Intercept: ", linear_reg.intercept_)

print("Coefficient: " , linear_reg.coef_)


# ### predictive model

# $$y = 1366427.618 - 87.309 \times Store + 101.841 \times Department + 705.852 \times Is Holiday+ 19.842 \times Temperature + 292.467 \times FuelPrice - 20.858 \times CPI - 183.231 \times Unemployment + 0.0859 \times Size + 0.0379 \times Markdowns - 677.895 \times Year + 130.009 \times Month - 15.400 \times Day $$

# ###  Most important feature is 'IsHoliday'
# ### meaning that whether that specific day is a holiday or not has the greatest impact on weekly sales
# ### Second most important is Fuel Price follwed by Unemployment
# ### least important is markdowns but a lot of the data was 0

# ### Testing and prediction

# In[8]:


y_pred = linear_reg.predict(X_test)
print(y_pred)


# ### RMSE

# In[9]:


from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
rmse = np.sqrt(mse)


# In[54]:


print(rmse)
