
# coding: utf-8

# In[37]:


# ### Importing the required packages and libraries
# # we will need numpy and pandas later
# import numpy as np
# import pandas as pd

# #import LogisticRegression Class
# from sklearn.linear_model import LogisticRegression
# #import RandomForestClassifier class
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import accuracy_score
# # Randomly splitting the original dataset into training set and testing set
# from sklearn.model_selection import train_test_split

#Really need these
import pandas as pd 
import numpy as np
from numpy import *


#Handy for debugging
import gc
import time
import warnings
import os

#Date stuff
from datetime import datetime
from datetime import timedelta

#Do some statistics
from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss
import math



#Machine learning tools
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy import sparse


## Performance measures
from sklearn.metrics import mean_squared_error


# In[38]:


# # reading a CSV file directly from Web, and store it in a pandas DataFrame:
# # "read_csv" is a pandas function to read csv files from web or local device:
# walmart_features_df =  pd.read_csv('features.csv')

# walmart_sampleSubmission_df =  pd.read_csv('sampleSubmission.csv')

# walmart_stores_df =  pd.read_csv('stores.csv')

# walmart_test_df =  pd.read_csv('test.csv')

# walmart_train_df =  pd.read_csv('train.csv')


# In[39]:


df = pd.read_csv('../data/merged-train-data.csv')
df[0::100000]


# In[40]:


#split with method
y = df['Weekly_Sales']
X = df.drop(columns=['Weekly_Sales','Type'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=2)

print(X.head())

print(y.head())


# In[41]:


# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(X_train.head())


# In[46]:


from sklearn.metrics import r2_score

num_est = [2,4,10,20,30,40,50,60,70,80,90,100]
acc_scores = []
rmse_scores = []
results = pd.DataFrame()

for i in num_est:
    my_RandomForest = RandomForestRegressor(n_estimators = i, bootstrap = True, random_state=3)
    my_RandomForest.fit(X_train,y_train)
    y_predict_rf = my_RandomForest.predict(X_test)
    y_predict_score = r2_score(y_test, y_predict_rf)
    mse = metrics.mean_squared_error(y_test, y_predict_rf)
    rmse = np.sqrt(mse)
    acc_scores.append(y_predict_score)
    rmse_scores.append(rmse)
    print('Processing with',i,'trees...')
    
results['Num of Trees'] = num_est
results['Accuracy Score'] = acc_scores
results['RMSE'] = rmse_scores
results


# I found that the best n_estimator to use was 60 by using the rmse and r2_score.
