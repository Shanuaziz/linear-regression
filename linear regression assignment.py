#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


df=pd.read_csv('car_age_price.csv')


# In[48]:


df.head()


# In[49]:


df.corr()


# In[79]:


#Separating the data into features and labels


# In[50]:


X=df[['Year']]
y=df['Price']


# In[80]:


#Dividing the dataset into test and train data


# In[51]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=44, shuffle=True)


# In[52]:


X.ndim


# In[53]:


y.ndim


# In[54]:


X.shape


# In[55]:


y.shape


# In[81]:


#Selecting the linear regression method from scikit-learn library


# In[56]:


from sklearn.linear_model import LinearRegression


# In[66]:


model = LinearRegression().fit(X_train, y_train)


# In[43]:


test_data=[[34.7]]
y_pred = model.predict(test_data)
y_pred


# In[82]:


#Validation


# In[42]:


from sklearn import metrics


# In[64]:


y_prediction = model.predict(X_train)
print("MAE on train data= " , metrics.mean_absolute_error(y_train, y_prediction))


# In[40]:


y_prediction = model.predict(X_test)
print("MAE on test data = " , metrics.mean_absolute_error(y_test, y_prediction))


# In[73]:


plt.scatter(X_train,y_train)
plt.plot(X_train, y_prediction ,color='b')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()


# In[78]:


test_data=[[2022]]
y_pred = model.predict(test_data)
y_pred


# In[ ]:




