
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import DatetimeIndex


# In[2]:


def mergeData(df):
    features = pd.read_csv('features.csv')
    storesdata = pd.read_csv('stores.csv')
    df = pd.merge(df, features, on=['Store','Date','IsHoliday'],how='inner')
    df = pd.merge(df, storesdata, on=['Store'], how='inner')
    return df


# In[3]:


merged_df = mergeData(pd.read_csv('train.csv'))


# In[4]:


merged_df.fillna(value=0, inplace=True)


# In[5]:


merged_df['Markdowns'] = merged_df['MarkDown1'] + merged_df['MarkDown2'] + merged_df['MarkDown3'] + merged_df['MarkDown4'] + merged_df['MarkDown5'] 
labelsToDrop = ['MarkDown1', 'MarkDown2', 'MarkDown3','MarkDown4','MarkDown5']
merged_df.drop(labels=labelsToDrop,axis=1, inplace=True)


# In[6]:


print(merged_df.head())


# In[7]:


df = merged_df
df.Date = pd.to_datetime(df.Date)
print(df.head())


# In[8]:


df['Year'] = DatetimeIndex(df['Date']).year
df['Month']= DatetimeIndex(df['Date']).month
df['Day'] = DatetimeIndex(df['Date']).day
df = df.drop(columns=['Date'])
df[0::10000]


# #### Exporting the merged data frame to a new csv file

# In[9]:


df.to_csv('merged-train-data.csv', index = False)

