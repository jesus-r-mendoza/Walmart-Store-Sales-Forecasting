
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
walmart_features_df =  pd.read_csv('../../data/features.csv')

walmart_sampleSubmission_df =  pd.read_csv('../../data/sampleSubmission.csv')

walmart_stores_df =  pd.read_csv('../../data/stores.csv')

walmart_test_df =  pd.read_csv('../../data/test.csv')

walmart_train_df =  pd.read_csv('../../data/train.csv')


# In[3]:


walmart_features_df.fillna(value=0, inplace=True)

#adding all the markdown sales and putting it in one column
walmart_features_df['Markdowns'] = walmart_features_df['MarkDown1'] + walmart_features_df['MarkDown2'] + walmart_features_df['MarkDown3'] + walmart_features_df['MarkDown4'] + walmart_features_df['MarkDown5']

labelsToDrop = ['MarkDown1', 'MarkDown2', 'MarkDown3','MarkDown4','MarkDown5']
walmart_features_df.drop(labels=labelsToDrop,axis=1, inplace=True)


# In[4]:


#merging datasets, joining walmart_Store_Df to features by the common column of store.
MergeFeatureAndStore_Df = pd.merge(walmart_features_df,
                 walmart_stores_df[['Store','Size']],
                 on='Store')

MergeFeatureAndStore_Df.head()


# In[5]:


#merging train dataset
Merged_Train_Features_Store_Df = pd.merge(MergeFeatureAndStore_Df, walmart_train_df[['Date','Dept','Weekly_Sales']], on='Date')
Merged_Train_Features_Store_Df.head()


# In[7]:


Merged_Train_Features_Store_Df.describe()


# In[6]:


#fix date into two columns one for month and other for year
from pandas import DatetimeIndex
df = Merged_Train_Features_Store_Df
df.Date = pd.to_datetime(df.Date)


# In[7]:


df['Year'] = DatetimeIndex(df['Date']).year
df['Month']= DatetimeIndex(df['Date']).month
df['Day'] = DatetimeIndex(df['Date']).day
df = df.drop(columns=['Date'])
df[0::1000]


# In[8]:


#creating the feature matrix
feature_cols = ['Store', 'Temperature','Fuel_Price','CPI','Unemployment', 'Markdowns', 'Size', 'Dept', 'Weekly_Sales', 'Year', 'Month', 'Day']
X = Merged_Train_Features_Store_Df[feature_cols]


# In[9]:


#Series of labels
y = Merged_Train_Features_Store_Df['IsHoliday']

y[0::10]


# In[10]:


Merged_Train_Features_Store_Df.info()


# In[11]:


#logreg instantiated as an object of LogisticRegression
logreg = LogisticRegression()


# In[12]:


#spliting the dataset, 30% for testing set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=10)


# In[16]:


print(X_train.shape)
print(y_train.shape)


print(X_test.shape)
print(y_test.shape)


# In[14]:


logreg.fit(X_train, y_train)


# In[17]:


y_predict_logreg = logreg.predict(X_test)


# In[18]:


score_logreg = accuracy_score(y_test, y_predict_logreg)


# In[19]:


print(score_logreg)


# # Cross-Validation
#

# In[20]:


from sklearn.model_selection import cross_val_score


# In[ ]:


#Applying 10-fold CV for logistic Regression

#creating the feature matrix
accuracy_list = cross_val_score(logreg, X, y, cv=5, scoring='accuracy')

print(accuracy_list)
