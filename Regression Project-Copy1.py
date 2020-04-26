
# coding: utf-8

# In[2]:


#importing dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston


# In[3]:


#understanding tne dataset
boston = load_boston()
print(boston.DESCR)


# In[5]:


# access data attributes
dataset = boston.data
for name, index in enumerate(boston.feature_names):
    print(index,name)


# In[6]:


# reshaping data
data = dataset[:,12].reshape(-1,1)


# In[7]:


#shape of the data
np.shape(dataset)


# In[8]:


# target values
target = boston.target.reshape(-1,1)


# In[9]:


# shape of target
np.shape(target)


# In[10]:


# ensuring that matplotlib is working
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color = 'green')
plt.xlabel('Lower income population')
plt.ylabel('Cost of House')
plt.show()


# In[11]:


# regression
from sklearn.linear_model import LinearRegression

# creating a regression model
reg = LinearRegression()

# fit the model
reg.fit(data, target)
 


# In[12]:


# prediction 
pred = reg.predict(data)


# In[16]:


# ensuring that matplotlib is working
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color = 'red')
plt.plot(data, pred, color = 'green')
plt.xlabel('Lower income population')
plt.ylabel('Cost of House')
plt.show()


# In[18]:


# circumventing curve using polinomial model
from sklearn.preprocessing import PolynomialFeatures


# to allow merging of models
from sklearn.pipeline import make_pipeline


# In[19]:


model = make_pipeline(PolynomialFeatures(3),reg)


# In[20]:


model.fit(data, target)


# In[21]:


pred = model.predict(data)


# In[22]:


# ensuring that matplotlib is working
get_ipython().run_line_magic('matplotlib', 'inline')
plt.scatter(data, target, color = 'red')
plt.plot(data, pred, color = 'green')
plt.xlabel('Lower income population')
plt.ylabel('Cost of House')
plt.show()


# In[23]:


# r_2 metric
from sklearn.metrics import r2_score


# In[24]:


# predict
r2_score(pred,target)

