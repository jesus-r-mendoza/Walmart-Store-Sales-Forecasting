
# coding: utf-8

# In[1]:


### Importing the required packages and libraries
# we will need numpy and pandas later
import numpy as np
import pandas as pd

#import LogisticRegression Class
from sklearn.linear_model import LogisticRegression
#import DecisionTreeClassifier class
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Randomly splitting the original dataset into training set and testing set
from sklearn.model_selection import train_test_split


# In[2]:


# reading a CSV file directly from Web, and store it in a pandas DataFrame:
# "read_csv" is a pandas function to read csv files from web or local device:
walmart_features_df =  pd.read_csv('features.csv')

walmart_sampleSubmission_df =  pd.read_csv('sampleSubmission.csv')

walmart_stores_df =  pd.read_csv('stores.csv')

walmart_test_df =  pd.read_csv('test.csv')

walmart_train_df =  pd.read_csv('train.csv')


# In[3]:


print("Feature Data Frame Info", walmart_features_df.info())

print("\n")

print("Store Data Frame Info", walmart_stores_df.info())

print("\n")

print("Test Data Frame Info",walmart_test_df.info())

print("\n")

print("Train Data Frame Info", walmart_train_df.info())


# In[4]:


#replace all the nan values with 0, also inplace = true make is permanent
walmart_features_df.fillna(value=0, inplace=True)


# In[5]:


#adding all the markdown sales and putting it in one column
walmart_features_df['Markdowns'] = walmart_features_df['MarkDown1'] + walmart_features_df['MarkDown2'] + walmart_features_df['MarkDown3'] + walmart_features_df['MarkDown4'] + walmart_features_df['MarkDown5'] 


# In[6]:


#dropping the unncessary columns,
labelsToDrop = ['MarkDown1', 'MarkDown2', 'MarkDown3','MarkDown4','MarkDown5']
walmart_features_df.drop(labels=labelsToDrop,axis=1, inplace=True)


# In[7]:


walmart_features_df.head()


# In[55]:


#merging datasets, joining walmart_Store_Df to features by the common column of store.
MergeFeatureAndStore_Df = pd.merge(walmart_features_df,
                 walmart_stores_df[['Store','Size']],
                 on='Store')


# In[56]:


MergeFeatureAndStore_Df


# In[62]:


#merging train dataset
Merged_Train_Features_Store_Df = pd.merge(MergeFeatureAndStore_Df, walmart_train_df[['Date','Dept','Weekly_Sales']], on='Date')


# In[63]:


Merged_Train_Features_Store_Df.head()


# In[64]:


Merged_Train_Features_Store_Df.info()


# In[60]:


Merged_Train_Features_Store_Df


# In[61]:


walmart_train_df

