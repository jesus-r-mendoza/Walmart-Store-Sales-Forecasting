
# coding: utf-8

# In[15]:


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


import matplotlib.pyplot as plt
import seaborn as sns


# In[16]:


def mergeData(df):
    features =pd.read_csv('features.csv')
    storesdata =pd.read_csv('stores.csv')
    df = pd.merge(df, features, on=['Store','Date','IsHoliday'],how='inner')
    df = pd.merge(df, storesdata, on=['Store'], how='inner')
    return df


# In[17]:


merged_df = mergeData(pd.read_csv('train.csv'))


# In[70]:


#merged_df.fillna(value=0, inplace=True)
#merged_df.dropna(inplace=True)


# In[18]:


merged_df.head()


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


def scatterplots(dataset,label):
    plt.figure()
    y = merged_df['Weekly_Sales']
    plt.scatter(dataset[label],y)
    plt.ylabel('Weekly_Sales')
    plt.xlabel(label)
    


# In[21]:


scatterplots(merged_df, 'Store')
scatterplots(merged_df, 'CPI')
scatterplots(merged_df, 'Temperature')
scatterplots(merged_df, 'Size')
scatterplots(merged_df, 'Fuel_Price')
scatterplots(merged_df, 'Dept')
scatterplots(merged_df, 'IsHoliday')


# In[22]:


#See what our data actually looks like with the describe function. 
merged_df.describe()


# In[23]:


merged_df.loc[merged_df['Weekly_Sales'] >350000,"Date"].value_counts()


# In[24]:


merged_df.columns


# In[25]:


merged_df.fillna(value=0, inplace=True)
#merged_df.dropna(inplace=True)


# In[26]:


merged_df['Markdowns'] = merged_df['MarkDown1'] + merged_df['MarkDown2'] + merged_df['MarkDown3'] + merged_df['MarkDown4'] + merged_df['MarkDown5'] 
labelsToDrop = ['MarkDown1', 'MarkDown2', 'MarkDown3','MarkDown4','MarkDown5']
merged_df.drop(labels=labelsToDrop,axis=1, inplace=True)


# In[27]:


sns.pairplot(merged_df)


# In[28]:


print(merged_df.head())


# In[29]:


from pandas import DatetimeIndex
df = merged_df
df.Date = pd.to_datetime(df.Date)
print(df.head())


# In[30]:


df['Year'] = DatetimeIndex(df['Date']).year
df['Month']= DatetimeIndex(df['Date']).month
df['Day'] = DatetimeIndex(df['Date']).day
df = df.drop(columns=['Date'])
df[0::1000]


# In[31]:


df.describe()


# In[41]:


print(merged_df.loc[merged_df['Weekly_Sales'] >350000,"Month"].value_counts())


# # Testing and Training Data

# In[155]:


df.columns


# In[156]:


y = df['Weekly_Sales']


# In[157]:


X = df[['Store', 'Dept','IsHoliday', 'Temperature','Fuel_Price', 'CPI', 'Unemployment', 'Size', 'Markdowns']]


# In[35]:


#here we need this library to scale X
#from sklearn import preprocessing

#normalizing feature comlumns 
#X = preprocessing.scale(X)


# In[158]:


#Randomly Splitting the original dataset into training set and testing set. 30% of data samples for testing, and rest 70% for training.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=101)


# In[159]:


lr = LinearRegression()


# In[160]:


lr.fit(X_train, y_train)


# # Print out Coefficients of the model

# In[161]:


lr.coef_


# In[162]:


predictions = lr.predict(X_test)


# In[163]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test (True Values)')
plt.ylabel('Predicted Values')


# # Evaulating the Model 
# 
# Calculating Mean Absolute Error, Mean Sqaure Error, And Root Mean Sqaure Error.

# In[164]:


from sklearn import metrics


# In[165]:


print('MAE ', metrics.mean_absolute_error(y_test,predictions))
print('MSE ', metrics.mean_squared_error(y_test,predictions))
print('RMSE ', np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# # RECREATE THE DATAFRAME 

# In[90]:


cdf = pd.DataFrame(lr.coef_, columns=['Coeff'])


# In[91]:


cdf

