
# coding: utf-8

# # brian canela

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
#walmart_features_df =  pd.read_csv('features.csv')

#walmart_sampleSubmission_df =  pd.read_csv('sampleSubmission.csv')

#walmart_stores_df =  pd.read_csv('stores.csv')

#walmart_test_df =  pd.read_csv('test.csv')

#walmart_train_df =  pd.read_csv('train.csv')


# In[3]:


df = pd.read_csv('../data/merged-train-data.csv')
df[0::100000]


# # Doing Logisitc Regression

# In[4]:


#creating the feature matrix 
feature_cols = ['Store', 'Temperature','Fuel_Price','CPI','Unemployment', 'Markdowns', 'Size', 'Dept', 'Weekly_Sales', 'Year', 'Month', 'Day']
X = df[feature_cols]


# In[5]:


#Series of labels
y = df['IsHoliday']
y[0::10]


# In[6]:


#logreg instantiated as an object of LogisticRegression
logreg = LogisticRegression()


# In[7]:


#Randomly Splitting the original dataset into training set and testing set. 30% of data samples for testing, and rest 70% for training.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=2)


# In[8]:


print(X_train.shape)
print(y_train.shape)


# In[9]:


print(X_test.shape)
print(y_test.shape)


# In[10]:


logreg.fit(X_train, y_train)


# In[11]:


y_predict_logreg = logreg.predict(X_test)
print(y_predict_logreg)


# # Accuracy Evaluation Using Logistic Regression

# In[12]:


score_logreg = accuracy_score(y_test, y_predict_logreg)
print(score_logreg)


# # # Cross-Validation

# In[13]:


from sklearn.model_selection import cross_val_score


# In[ ]:


#Applying 10-fold CV for logistic Regression 

#creating the feature matrix 
accuracy_cv_list = cross_val_score(logreg, X, y, cv=10, scoring='accuracy')

print(accuracy_cv_list)


# In[ ]:


accuracy_list = accuracy_cv_list.mean()


# In[ ]:


print(accuracy_list)


#  

# In[ ]:


df.head()


# In[ ]:


def categorical_to_numeric(x):
    if x == False:
        return 0
    elif x == True:
        return 1


# In[ ]:


df['IsHoliday'] = df['IsHoliday'].apply(categorical_to_numeric)


# In[ ]:


df.head()


# In[ ]:


#creating the feature matrix 
feature_cols_ = ['Store','Dept','Weekly_Sales','IsHoliday','Temperature','Fuel_Price','CPI','Unemployment','Type','Size','Markdowns','Year','Month','Day']
X = df[feature_cols]


# In[ ]:


#Series of labels
y = df['IsHoliday']
y[0::10]


# In[ ]:


#logreg instantiated as an object of LogisticRegression
lr = LogisticRegression()


# In[ ]:


#Randomly Splitting the original dataset into training set and testing set. 30% of data samples for testing, and rest 70% for training.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=2)


# In[ ]:


lr.fit(X_train, y_train)


# # ESTIMATING THE PROBABILITY OF EVENT THAT ITS A HOLIDAY

# In[ ]:


##
y_predict_prob_lr = logreg.predict_proba(X_test)


# In[ ]:


#predicting the estimated likelihood of both label for testing sets
print(y_predict_prob_lr)

print("\n")

#line prints the estimated likelihood of label=1" for testing set
print(y_predict_prob_lr[:,1])
print("\n")

#predicts the actual label of the testing set
print(y_test)
print("\n")

#line prints the actual label of the testing set
print(y_predict_logreg)


# In[ ]:


from sklearn import metrics
#check dataset, predict whatever is dataset, 
#the function moves threadshold and detects , remember change pos_label to whatever it is, pos_leabel is define by u, so this 
#pos_label=1 this is for prediciting isHolday is true!
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict_prob_lr[:,1], pos_label=1)

print(fpr) # false alarm

print(tpr) #true positive


# In[ ]:


#auc
AUC = metrics.auc(fpr, tpr)
print("Area Under the Curve: ", AUC)


# In[ ]:


#ROC CURVE
# Importing the "pyplot" package of "matplotlib" library of python to generate 
# graphs and plot curves:
import matplotlib.pyplot as plt

# The following line will tell Jupyter Notebook to keep the figures inside the explorer page 
# rather than openng a new figure window:
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure()

# Roc Curve:
plt.plot(fpr, tpr, color='black', lw=2, 
     label='ROC Curve (area = %0.2f)' % AUC)

# Random Guess line:
plt.plot([0, 1], [0, 1], color='blue', lw=1, linestyle='--')

# Defining The Range of X-Axis and Y-Axis:
plt.xlim([-0.005, 1.005])
plt.ylim([0.0, 1.01])

# Labels, Title, Legend:
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC)')
plt.legend(loc="lower right")

plt.show()

