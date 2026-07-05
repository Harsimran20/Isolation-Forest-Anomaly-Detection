#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn. preprocessing import StandardScaler


# In[8]:


df = pd.read_csv("thyroid_dataset.csv")


# In[14]:


df.shape


# In[20]:


X = df.drop("Outlier_label",axis = 1)
y = df["Outlier_label"]


# In[22]:


X.head()


# In[24]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[26]:


from sklearn.ensemble import IsolationForest
clf = IsolationForest(n_estimators = 200, contamination = "auto",random_state = 42)


# In[28]:


labels = clf.fit_predict(X_scaled)


# In[30]:


labels


# In[32]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X_scaled)


# In[34]:


plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0],X_pca[:,1], c=labels)
plt.xlabel("PC1")
plt.ylabel("PC2")


# In[38]:


import numpy as np
n_outliers = np.sum(labels == -1)
n_normal = np.sum(labels == 1)
print("Outliers: ",n_outliers)
print("Normal: ",n_normal)


# In[ ]:




