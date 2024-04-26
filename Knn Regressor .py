#!/usr/bin/env python
# coding: utf-8

# In[2]:


## KNN Regressor 


# In[3]:


from sklearn.datasets import make_regression
x,y=make_regression(n_samples=1000,n_features=2,noise=10,random_state=42)


# In[4]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
   x,y,test_size=0.33,random_state=42)


# In[5]:


from sklearn.neighbors import KNeighborsRegressor


# In[6]:


regressor=KNeighborsRegressor(n_neighbors=5,algorithm="auto")
regressor.fit(x_train,y_train)


# In[7]:


y_pred=regressor.predict(x_test)


# In[8]:


from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[9]:


print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




