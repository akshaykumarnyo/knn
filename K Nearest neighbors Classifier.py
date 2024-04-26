#!/usr/bin/env python
# coding: utf-8

# In[2]:


## knn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


from sklearn.datasets import make_classification
x,y=make_classification(
   n_samples=1000, ## 1000 observations
   n_features=3, ## 3 total features
   n_redundant=1,
   n_classes=2, ## binary target/Label
   random_state=999
)


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)


# In[12]:


x_train


# In[13]:


x_test


# In[15]:


from sklearn.neighbors import KNeighborsClassifier


# In[16]:


classifier=KNeighborsClassifier(n_neighbors=5,algorithm="auto")


# In[17]:


classifier.fit(x_train,y_train)


# In[18]:


y_pred=classifier.predict(x_test)


# In[19]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report


# In[21]:


print(confusion_matrix(y_pred,y_test))
print(accuracy_score(y_pred,y_test))
print(classification_report(y_pred,y_test))


# In[28]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

for i in range(1, 11):
    classifier = KNeighborsClassifier(n_neighbors=i, algorithm="auto")
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(f"accuracy_score for n_neighbors={i}:\n{accuracy_score(y_pred, y_test)}")
    print(f"Confusion matrix for n_neighbors={i}:\n{confusion_matrix(y_pred, y_test)}")
    print(f"classification_report for n_neighbors={i}:\n{classification_report(y_pred, y_test)}")


# In[35]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report



classifier= KNeighborsClassifier(n_neighbors=7, algorithm="auto")
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(f"accuracy_score for n_neighbors=7:\n{accuracy_score(y_pred, y_test)}")
print(f"Confusion matrix for n_neighbors=7:\n{confusion_matrix(y_pred, y_test)}")
print(f"classification_report for n_neighbors=7:\n{classification_report(y_pred, y_test)}")


# In[29]:


## Hiperparameter tunnning


# In[39]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'n_neighbors': range(1, 21)}  # Range of k values to try

# Create a KNN classifier
knn = KNeighborsClassifier()

# Instantiate the GridSearchCV object
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

# Perform hyperparameter tuning
grid_search.fit(x_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
accuracy = best_model.score(x_test, y_test)
print("Accuracy on Test Set:", accuracy)


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




