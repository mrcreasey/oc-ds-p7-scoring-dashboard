import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image

im = Image.open("favicon.ico")
st.set_page_config(
    page_icon=im
)

st.title('Scoring Dashboard')
st.header('OpenClassrooms Projet 7')
st.subheader('Parcours Data Scientist')
st.markdown("<i>Modèle de Scoring</i>", unsafe_allow_html=True)

st.sidebar.subheader("Menu")


# plotting seaborn
fig, ax = plt.subplots()
df_col=pd.DataFrame({
    'y_true':[0,1,0,1,0,0,1,0,0,1,1,1,0,0,0],
    'y_pred':[0,0,0,0,1,0,1,1,0,1,0,1,1,0,0],
})
sns.heatmap(df_col.corr(), ax=ax)
st.write(fig)