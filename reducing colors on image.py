#!/usr/bin/env python
# coding: utf-8

# # reducing colors on image
# 

# "import matplotlib.pyplot as plt" this command is used for matploting and ".pyplot" for ploting in python

# In[26]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# "pip install scikit-image" is used  installing the scikit-image package and it is providing the functions and algorithms provided by scikit-image in  Python programming.

# In[27]:


pip install scikit-image


# in scikit image packaging we importing the read imaging and show image commands so this are given below

# In[28]:


from skimage.io import imread,imshow


# In[29]:


image = imread('siddharth.jpg')


# In[30]:


imshow(image)


# KMeans is a part of the Scikit-learn (sklearn) library it is in popular machine learning library in Python used for various tasks such as classification, regression, and clustering but mostly clustering. 

# In[31]:


from sklearn.cluster import KMeans


# its providing information about image shape .

# In[32]:


image.shape


# we get reshape the image

# In[33]:


X = image.reshape(-1,3)/255


# In KMeans we using clustering method.

# In[34]:


kmeans = KMeans(n_clusters=3)


# In[35]:


X.shape


# we using "Kmean.fit(x) for fiting of kn part in "X" so "X" is the image 

# In[36]:


kmeans.fit(X)


# In[37]:


kmeans.cluster_centers_


# In[38]:


kmeans.labels_


# In[39]:


new_X = kmeans.cluster_centers_[kmeans.labels_]


# kmeans providing the reshape of image and reducing colors on image

# In[40]:


image = new_X.reshape(1280, 1229, 3)


# In[41]:


imshow(image)

