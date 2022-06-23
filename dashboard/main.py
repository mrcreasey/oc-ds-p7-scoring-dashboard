import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from lightgbm import LGBMClassifier
import requests

# -------------------------------------------
# SECRETS
# stored locally in .streamlit/secrets.toml
# pasted into configuration in streamlit console when deploying
# -------------------------------------------
# Everything is accessible via the st.secrets dict (subsection [config]) :

API_URL=st.secrets['config']['API_URL']
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

client_id=None
threshold=0.542

def main():
    """Display data"""
    global client_id
    global threshold
    list_clients=get_list_clients()
    nb_clients=len(list_clients)
    client_id= st.sidebar.selectbox(f'Choisir un client (count={nb_clients}) :',list_clients)
    st.write('Selected client ', client_id)
    pred_data = get_client_predict(client_id,threshold,return_data=True)
    df_client=None
    if isinstance(pred_data,dict):
        refuse_loan= pred_data.get('y_pred',1)
        loan='prêt approuvé' if refuse_loan==0 else 'prêt refusé'
        proba=pred_data.get('y_pred_proba',-1)
        st.write(f'{loan} (probabilité = {proba:.3f}); threshold = {threshold:.3f}')
        st.slider(label='probabilité',min_value=0., max_value=1., value=proba)
    # df_client=get_client_data(client_id)
        client_data= pred_data.get('client_data',{})
        df_client =pd.DataFrame.from_dict(client_data, orient='index')
    if isinstance(df_client, pd.DataFrame):
        st.dataframe(df_client)

# ------------------------------------------------
# Requests to API server
# ------------------------------------------------

@st.cache
def get_list_clients():
    """Load list of clients""" 
    response = requests.get(f'{API_URL}/clients')
    data = response.json()
    list_clients=list(data)
    return list_clients

@st.cache
def get_client_data(id):
    response = requests.get(f'{API_URL}/client/{id}')
    data = response.json()
    if data.get('error'):
        st.write(data)
    else:
        return pd.DataFrame.from_dict(data, orient='index')

@st.cache
def get_client_predict(id, threshold=None, return_data=False):
    """predict give loan or not"""
    params=dict(return_data=return_data)
    if not threshold is None:
        params['threshold']= threshold
    response = requests.get(f'{API_URL}/predict/{id}', params=params)
    data = response.json()
    if data.get('error'):
        st.write(data)
    else:
        return data
# ------------------------------------------------
# Plotting routines
# ------------------------------------------------

def plot_seaborn():
    """test plotting seaborn"""
    fig, ax = plt.subplots()
    df_col=pd.DataFrame({
        'y_true':[0,1,0,1,0,0,1,0,0,1,1,1,0,0,0],
        'y_pred':[0,0,0,0,1,0,1,1,0,1,0,1,1,0,0],
    })
    sns.heatmap(df_col.corr(), ax=ax)
    st.write(fig)

main()