import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Example page icons 💶💰💸💳🪙🤑💲
st.set_page_config(
    page_icon='💶'
)

st.title('Scoring Dashboard')
st.header('OpenClassrooms Projet 7')
st.subheader('Parcours Data Scientist')
st.markdown("<i>Modèle de Scoring</i>", unsafe_allow_html=True)

st.sidebar.subheader("Menu")
list_clients=[12334,5663,34456,8877]
st.sidebar.selectbox('Choisir un client',list_clients)


# plotting seaborn
fig, ax = plt.subplots()
df_col=pd.DataFrame({
    'y_true':[0,1,0,1,0,0,1,0,0,1,1,1,0,0,0],
    'y_pred':[0,0,0,0,1,0,1,1,0,1,0,1,1,0,0],
})
sns.heatmap(df_col.corr(), ax=ax)
st.write(fig)