import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from lightgbm import LGBMClassifier

# -------------------------------------------
# SECRETS
# stored locally in .streamlit/secrets.toml
# pasted into configuration in streamlit console when deploying
# -------------------------------------------
# Everything is accessible via the st.secrets dict:
model_server=st.secrets['config']['MODEL_SERVER']
model_file=st.secrets['config']['MODEL_FILE']
data_server=st.secrets['config']['DATA_SERVER']
data_file=st.secrets['config']['DATA_FILE']
default_threshold=st.secrets['config']['THRESHOLD']

model_path=f'{model_server}/{model_file}'
data_path=f'{data_server}/{data_file}'

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

model: LGBMClassifier = load_pickle(model_path)



# -------------------------------------------
#  PAGE LAYOUT
# Example page icons 💶💰💸💳🪙🤑💲
st.set_page_config(
    page_title='Scoring Dashboard',
    page_icon='💶',
    initial_sidebar_state="expanded",
    layout="wide",
)

st.title('Scoring Dashboard')
st.header('OpenClassrooms Projet 7')
st.subheader('Parcours Data Scientist')
st.markdown("<i>Modèle de Scoring</i>", unsafe_allow_html=True)

st.sidebar.subheader("Menu")

@st.cache
def load_model():
    """Load Model for developing dashboard (replace with API afterwords)"""
    model:LGBMClassifier=load_pickle(model_path)
    return model

@st.cache
def load_clients_data():
    """Load list of clients"""
    data:pd.DataFrame = load_pickle(data_path)
    if len(data)>50:
        data=data.head(50)
    # data.index should already have been set to SK_ID_CURR
    # this is so we do not have to drop the column before making predictions
    if 'SK_ID_CURR' in data.columns:
        data=data.set_index('SK_ID_CURR')
    return data

def get_list_clients():
    # Load client data
    data:pd.DataFrame = load_clients_data()
    list_clients=list(data.index)
    return list_clients

list_clients=get_list_clients()
st.sidebar.selectbox('Choisir un client',list_clients)


# plotting seaborn
fig, ax = plt.subplots()
df_col=pd.DataFrame({
    'y_true':[0,1,0,1,0,0,1,0,0,1,1,1,0,0,0],
    'y_pred':[0,0,0,0,1,0,1,1,0,1,0,1,1,0,0],
})
sns.heatmap(df_col.corr(), ax=ax)
st.write(fig)

