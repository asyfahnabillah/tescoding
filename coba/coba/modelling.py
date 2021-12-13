#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Import Library
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# me-non aktifkan peringatan pada python
import warnings 
warnings.filterwarnings('ignore')


# In[10]:


df=pd.read_csv("df.csv")
df


# In[11]:


df.head(10)


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


df.isnull().sum()


# In[15]:


df.shape


# In[16]:


#Mengetahui bentuk data yang ada di dalam kolom Perkerjaan
df['PEKERJAAN'].unique()


# In[17]:


#Mengetahui jumlah data yang ada di dalam kolom Perkerjaan
len(df['PEKERJAAN'].unique())


# In[18]:


#Mengetahui kunci (keys) elemen di dalam dataset
df.keys()


# In[19]:


#Memvisualisasikan data
#Menampilkan visualisasi bar plot dan pie chart untuk menghitung frekuensi dari kolom Di Diagnosis hipertensi dalam dataset data
f,ax=plt.subplots(1,2,figsize=(18,8))
df['DI DIAGNOSIS HIPERTENSI'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_xlabel('DI DIAGNOSIS HIPERTENSI')
ax[0].set_ylabel('JUMLAH')
sns.countplot('DI DIAGNOSIS HIPERTENSI',data=df,ax=ax[1])
ax[1].set_title('DI DIAGNOSIS HIPERTENSI')


# In[20]:



#Metrics
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report

# Import libarary confusion matrix
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

#Model Select
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.model_selection import train_test_split
# Import libarary Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB


# In[21]:


X=df.iloc[:, :-1].values
y=df.iloc[:, -1:].values


# In[22]:


print (X)


# In[23]:


print (y)


# In[24]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,0] = le.fit_transform(X[:,0])


# In[25]:


le = LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])


# In[26]:


le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])


# In[27]:


le = LabelEncoder()
X[:,3] = le.fit_transform(X[:,3])


# In[28]:


le = LabelEncoder()
X[:,4] = le.fit_transform(X[:,4])


# In[29]:


le = LabelEncoder()
X[:,5] = le.fit_transform(X[:,5])


# In[30]:


le = LabelEncoder()
X[:,6] = le.fit_transform(X[:,6])


# In[31]:


le = LabelEncoder()
X[:,7] = le.fit_transform(X[:,7])


# In[32]:


le = LabelEncoder()
X[:,8] = le.fit_transform(X[:,8])


# In[33]:


print (X)


# In[44]:


print(y)


# In[34]:


le = LabelEncoder()
y = le.fit_transform(y)


# In[35]:


X_train,X_test,y_train,y_test =train_test_split(X,y,train_size=0.8, test_size=0.2,random_state=0)


# In[36]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[37]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[38]:


print (X_train)


# In[39]:


gaussian = GaussianNB()
gaussian.fit(X_train, y_train)


# In[40]:


model = GaussianNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)

