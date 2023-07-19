#!/usr/bin/env python
# coding: utf-8

# Data Exploration:
# 
# Perform descriptive analysis. Understand the variables and their corresponding values. On the columns below, a value of zero does not make sense and thus indicates missing value:
# 
# Glucose
# 
# BloodPressure
# 
# SkinThickness
# 
# Insulin
# 
# BMI
# 
# Visually explore these variables using histograms. Treat the missing values accordingly.
# 
# There are integer and float data type variables in this dataset. Create a count (frequency) plot describing the data types and the count of variables. 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print('All library imported')


# In[2]:


#load the data
data=pd.read_csv('health care diabetes.csv')
print ('data loaded')


# In[3]:


data.head()


# In[4]:


#shape of data
data.shape


# In[5]:


#missing values
data.isnull().sum()


# In[6]:


data.info()


# In[7]:


#know the target data 
data['Outcome'].value_counts()


# In[8]:


#create histogram distribution of the data
plt.hist(data['Glucose'])
plt.show()


# In[9]:


data[(data['Glucose']==0)].shape


# In[10]:



data['Glucose'].mean()


# In[11]:


#fill these zeros
data.loc[data['Glucose']==0,'Glucose']=120.8945312


# In[12]:


data[(data['Glucose']==0)].shape


# In[13]:


#create histogram distribution of the data
plt.hist(data['BloodPressure'])
plt.show()


# In[14]:


#create histogram distribution of the data
plt.hist(data['SkinThickness'])
plt.show()


# In[15]:


# data is not normally distribute. right skeweness right side long tail.
plt.hist(data['Insulin'])
plt.show()


# In[16]:


plt.hist(data['BMI'])
plt.show()


# In[17]:


plt.hist(data['Age'])
plt.show()


# In[18]:


plt.hist(data['DiabetesPedigreeFunction'])
plt.show()


# In[19]:


data.describe().T # T transpose


# In[20]:


#data exploration
variables = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] 
for i in variables: 
    data[i].replace('0',np.nan) 
    data[i].fillna(data[i].median(), inplace=True) 


# In[21]:


data.head()


# In[22]:


variables = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for i in variables:
    #data[i].replace(0,np.nan)
    data[i].replace(0,data[i].median(), inplace=True)


# In[23]:


data.head()


# #plotting count
# #create satter chart
# #perform correlation analysis

# In[24]:


data.columns


# In[25]:


plt.figure(figsize=(12,12))
sns.countplot(x=data['Age'],hue='Outcome',data=data)


# In[26]:


dib_person=data[data['Outcome']==1]


# In[27]:


dib_person


# In[28]:


sns.histplot(x=dib_person['Glucose'])
plt.show()


# In[29]:


dib_person['Glucose'].value_counts().head(10)


# In[30]:


sns.histplot(x=dib_person['BloodPressure'])
plt.show()


# In[31]:


dib_person['BloodPressure'].value_counts().head(10)


# In[32]:


#scatter plot to find data who have diabetic
plt.scatter(x=dib_person['BloodPressure'],y=dib_person['Glucose'])
plt.xlabel('BloodPressure')
plt.ylabel('Glucose')


# In[33]:


sns.scatterplot(x='Glucose',y='BloodPressure',hue='Outcome',data=data)


# In[34]:


sns.scatterplot(x='SkinThickness',y='Insulin',hue='Outcome',data=data)


# In[35]:


#correlation analysis.Visual
data.corr()


# In[68]:


plt.figure(figsize=(5,5))
sns.heatmap(data.corr(),annot=True,cmap='viridis')


# #DATA MODELLING

# In[37]:


#DATA preporcocessing
X=data.iloc[:,:-1].values


# In[38]:


X


# In[39]:


y=data.iloc[:,-1].values


# In[40]:


y


# In[41]:


#train set test tst split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)


# In[42]:


X_train.shape


# In[43]:


X_test.shape


# In[44]:


import warnings
warnings.filterwarnings('ignore')


# #1.Logistic regression

# In[45]:


from sklearn.linear_model import LogisticRegression #LR is Class so we create object
model1=LogisticRegression()


# In[46]:


#training
model1.fit(X_train,y_train)#training fit an be used


# In[47]:


y_pred1=model1.predict(X_test)


# In[48]:


#train score & test score
print('Train score',model1.score(X_train,y_train))
print('Test score',model1.score(X_test,y_test))


# In[49]:


from sklearn.metrics import confusion_matrix,classification_report


# In[52]:


print(confusion_matrix(y_test,y_pred1))
# 0      1 (output)
#0  TN    FP
#1    FN    TP
# recall formula=TP/TP+FN


# In[53]:


print(classification_report(y_test,y_pred1))


# In[58]:


#Prepare ROC curve
from sklearn.metrics import roc_auc_score,roc_curve
prob=model1.predict_proba(X)
#prob
#select prob for the psitive outcome onnly
prob=prob[:,1]
#calculate area under the curve
auc=roc_auc_score(y,prob)
print('AUC score:',auc)


# In[61]:


#claculat roc curve 
fpr,tpr,thresholds=roc_curve(y,prob)
#plot 
plt.plot([0,1],[0,1],linestyle='--')
plt.plot(fpr,tpr,marker='.')


# In[62]:


import joblib
joblib.dump(model1,'Logistic.pkl')
print('model1saved')


# In[64]:


#Load the model
Pred_model=joblib.load('Logistic.pkl')
print('model loaded')


# In[65]:


data.columns


# In[66]:


Pregnancies=2
Glucose=148
BloodPressure=72
SkinThickness=40
Insulin=100
BMI=25.5
DiabetesPedigreeFunction=0.35
Age=35
output=Pred_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
       BMI, DiabetesPedigreeFunction, Age]])
print('Person has',output)


# In[67]:


#another file you predict the model
import joblib
Pred_model=joblib.load('Logistic.pkl')
print('model loaded')


# 2.Decision Tree
# 3.Random forest
# 4.KNN 
# 5,SVM

# In[ ]:




