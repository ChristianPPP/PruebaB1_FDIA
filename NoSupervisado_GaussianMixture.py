#!/usr/bin/env python
# coding: utf-8

# In[0]
# Copyright (c) 2022.
# Realizado por: Arias-Chalá-Palacios
# All rights reserved.
# %matpotlib inline 

from matplotlib import pyplot as plt
import pandas as pd
data = pd.read_csv('./summer-products-with-rating-and-performance_2020-08.csv')
datac = [data['price'], data['rating']] 
data2 = pd.concat(datac, axis=1, join='inner')

# In[2]
# training gaussian mixture model 
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=5)
gmm.fit(data2)

# In[3]
#predictions from gmm
labels = gmm.predict(data2)
frame = pd.DataFrame(data2)
frame['cluster'] = labels
frame.columns = ['rating', 'price', 'cluster']

# In[4]
color=['blue','green','cyan', 'black', 'yellow']
for k in range(0,5):
    data = frame[frame["cluster"]==k]
    plt.scatter(data["rating"],data["price"],c=color[k])
plt.show()

# %%
