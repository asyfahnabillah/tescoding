#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import base64
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.figure_factory as ff



st.title('Analisis Hipertensi Menggunakan Algoritma Naïve Bayes') #Specify title of your app
st.write('data ini berasal dari Dinas Kesehatan Kabupaten Purwakarta .')
st.sidebar.markdown('## Data Import') #Streamlit also accepts markdown
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv") #data uploader

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.markdown('### Data Sample')
    st.write(data.head())
    id_col = st.sidebar.selectbox('Pick your ID column', options=data.columns)
    cat_features = st.sidebar.multiselect('Pick your categorical features', options=[c for c in data.columns], default = [v for v in data.select_dtypes(exclude=[int, float]).columns.values if v != id_col])
    clusters = data['cluster']
    df_p = data.drop(id_col, axis=1)
    if cat_features:
        df_p = pd.get_dummies(df_p, columns=cat_features) #OHE the categorical features
    prof = st.checkbox('Check to profile the classification')

else:
    st.markdown("""
    <br>
    <br>
    <br>
    <br>
    <br>
    <br>
    <h1 style="color:#26608e;"> Upload your CSV file to begin clustering </h1>
    <h3 style="color:#f68b28;"> Customer segmentation </h3>
    """, unsafe_allow_html=True) 

    st.markdown('<hr>', unsafe_allow_html=True)
st.markdown(f'<h3 "> Features Overview </h3>',
            unsafe_allow_html=True)



