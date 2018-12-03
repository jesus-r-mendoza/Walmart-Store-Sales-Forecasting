
# coding: utf-8

# # PCA - Feature Reduction

# In[1]:


import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
# Importing function to normalize the data
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
# importing the required module
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('../../data/merged-train-data.csv')
df[0::100000]


# In[3]:


feat = ['Store','Dept','IsHoliday','Temperature','Fuel_Price','CPI','Unemployment','Size','Markdowns','Year','Month','Day']
X = df[feat]
y = df['Weekly_Sales']


# In[19]:


def plot(x_coors, y_coors, y_axis_name):

    plt.plot(x_coors, y_coors, color='green', marker='o', markerfacecolor='red', markersize=6)

    # naming the x axis
    plt.xlabel(' # of PCA components ')
    # naming the y axis
    plt.ylabel(y_axis_name)

    # giving a title to my graph
    plt.title(y_axis_name + ' vs. PCA Components')

    # function to show the plot
    plt.show()


# # PCA on Linear Regression

# In[4]:


# Performing PCA
# Total of 12 features, testing all reductions

print('\n= = = = = = = = Linear Regrression = = = = = = = = ')

lin_x = []
lin_y = []

for i in range(1,13):

    pca = PCA(n_components=i, whiten='True')
    new_X = pca.fit(X).transform(X)
    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.3, random_state=2)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    print('\nFor', i, 'components, these are the results\n')
    print("   Intercept: ", lin_reg.intercept_)
    print("   Coefficient: " , lin_reg.coef_)

    y_pred = lin_reg.predict(X_test)
    mse = metrics.mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    print('   RMSE:',rmse)

    lin_x.append(i)
    lin_y.append(rmse)


# In[13]:


# Plotting the linear regression graph
plot(lin_x, lin_y, 'RMSE')


# This graph shows us that when the number of components for PCA is greater than 8, the RMSE does not change much. Therfore, we can conclude that 8 components for PCA is a good estimater.

# # PCA on Decision Tree Regression

# In[8]:


# Performing PCA
# Total of 12 features, testing all reductions

print('\n= = = = = = = = Decision Tree Regressor = = = = = = = = ')

dec_x = []
dec_y = []

for i in range(1,13):

    pca = PCA(n_components=i, whiten='True')
    new_X = pca.fit(X).transform(X)
    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.3, random_state=2)

    dec_tree_regressor = DecisionTreeRegressor()
    dec_tree_regressor.fit(X_train, y_train)

    score_dt = dec_tree_regressor.score(X_test, y_test)
    y_pred = dec_tree_regressor.predict(X_test)

    print('\nFor', i, 'components, these are the results\n')
    print('   Score:',score_dt)

    dec_x.append(i)
    dec_y.append(score_dt)


# In[18]:


# Plotting the decision tree graph
plot(dec_x, dec_y, 'Accuracy Score')


# Here we can see that, condensing the features from 12 to 8 features (using PCA) gives us the highest accuracy with the lowest number of features. Any less features than 8, would result in an 8 % drop in accuracy.

# # PCA on Random Forest

# In[15]:


print('\n= = = = = = = = Random Forest = = = = = = = = ')

forest_x = []
forest_y = []

for i in range(1,13,2):

    pca = PCA(n_components=i, whiten='True')
    new_X = pca.fit(X).transform(X)
    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.3, random_state=2)

    rand_fst = RandomForestRegressor(n_estimators = 60, bootstrap = True, random_state=3)
    rand_fst.fit(X_train,y_train)

    y_predict_rf = rand_fst.predict(X_test)
    y_predict_score = r2_score(y_test, y_predict_rf)
    mse = metrics.mean_squared_error(y_test, y_predict_rf)
    rmse = np.sqrt(mse)

    print('\nFor', i, 'components, these are the results\n')
    print('   RMSE:', rmse)

    forest_x.append(i)
    forest_y.append(rmse)



# In[17]:


# Plottinh the random forest graph
plot(forest_x, forest_y, 'RMSE')


# Here in this graph, for PCA on random forest, we can see that there is not much benefit to reducing the number of features. Especially when compared to the RSME values of using all 12 features (as in the other dedicated Random Forest ipynb file). Here, the lower values range around 5k, where as in the Random Forest file they are around 3k.
# * Note: we only performed PCA for every other value (1,3,5,7 ..) unlike the other sets. This is merely to save time.

# In[15]:


# Some statistical analysis of the data

l = len(df['Weekly_Sales'])
sm = 0
mx = 0
for i in df['Weekly_Sales']:
    if i > mx:
        mx = i
    sm += i

avg = sm/l

abvSum = 0
for i in df['Weekly_Sales']:
    if i > avg:
        abvSum  += 1

print('The maximum value for Weekly_Sales is:',mx)
print('The average Weekly_Sales are:',sm/l)
print('This is the amount of values that are over the average:',abvSum,'; From a total of:', l, 'values.')
