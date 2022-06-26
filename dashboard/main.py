import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import requests

# Type checking
from typing import Union
from api_models import ClientExplainResponse, ClientPredictResponse, ErrorResponse
from gauge import plot_gauge

# -------------------------------------------
# SECRETS
# stored locally in .streamlit/secrets.toml
# pasted into configuration in streamlit console when deploying
# -------------------------------------------
# Everything is accessible via the st.secrets dict (subsection [config]) :

API_URL=st.secrets['config']['API_URL']
default_threshold=st.secrets['config']['THRESHOLD']

# -------------------------------------------
#  PAGE LAYOUT
# Example page icons 💶💰💸💳🪙🤑💲
st.set_page_config(
    page_title='Scoring Dashboard',
    page_icon='💸',
    # initial_sidebar_state="expanded",
    layout="wide",
)


# -----------------------------------------------------
# Header
def show_header():
    html_header="""
        <head>
            <meta charset="utf-8">
            <meta name="author" content="Mark Creasey">
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>             
    """ 
    st.write(html_header,unsafe_allow_html=True)
    # --------------------------------------------------------------------
    # Logo + Title
    t1, t2 = st.columns((1,5)) 
    t1.image('dashboard/logo_projet_fintech.png', width = 120)
    c2= t2.container()
    c2.title('Scoring Dashboard')
    c2.subheader('OpenClassrooms Projet 7')
    c2.markdown('_Parcours Data Scientist_')


# ----------------------------------------------------
# Add custom styles
st.markdown(
    """<style>
    .streamlit-expanderHeader {font-size: x-large;background:#FFDDDD;margin-bottom:10px;}
    </style>""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# load JS visualization code to notebook. Without this, the SHAP plots won't be displayed
shap.initjs()        


# ----------------------------------------
def init_key(key,value):
    if key not in st.session_state:
        st.session_state[key] = value

def initialise_session_state():
    # init_key('client_id',None)
    init_key('proba',0)
    init_key('threshold',0.542)
    init_key('threshold100',st.session_state.threshold*100)

# ----------------------------------------
# Load list of clients
@st.cache
def get_list_clients():
    """API - load list of clients""" 
    response = requests.get(f'{API_URL}/clients')
    return list(response.json())
 
def show_select_client(col):
    """Selectionne un client parmi la liste des clients"""
    with st.spinner('recuperation de la liste de clients'):
        list_clients=get_list_clients()
        nb_clients=len(list_clients)
        # client_id= st.sidebar.selectbox(f'Choisir un client (count={nb_clients}) :',list_clients)
        init_key('client_id',list_clients[0])
        col.selectbox('Selectionne un client', list_clients, key='client_id', on_change=on_change_client, help='SK_ID_CURR')

def on_change_client():
    update_client_data()

# ----------------------------------------------
# Predict client data
@st.cache()
def get_client_data(id):
    response = requests.get(f'{API_URL}/client/{id}')
    data = response.json()
    if data.get('error'):
        st.write(data)
    else:
        return pd.DataFrame.from_dict(data, orient='index')

@st.cache()
def get_client_predict(id, threshold=None, return_data=False)->Union[ClientPredictResponse,None]:
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


def update_client_data():
    """Mettre à jour données du client et prediction de risque"""
    client_id= st.session_state.client_id
    threshold= None # st.session_state.threshold
    pred_data = get_client_predict(client_id,threshold,return_data=True)
    if isinstance(pred_data,dict):
        st.session_state.proba=pred_data.get('y_pred_proba')
        st.session_state.client_data=series_from_dictkey(pred_data,'client_data')


def on_change_threshold():
    print(f'on_change_threshold')
    st.session_state.threshold =st.session_state.threshold100/100

def show_threshold_slider(col):
    col.slider('Threshold',0.,100.,1.,format='%g %%', 
    key='threshold100',on_change=on_change_threshold)


def show_metrics(mc):
    '''Predict, et retourner les données client'''
    m2,m3,m4= mc.columns(3)
    m2.metric(label =f'Décision',
    value = 'accepté' if (st.session_state.proba < st.session_state.threshold) else 'refusé',
     delta = f'threshold = {st.session_state.threshold100:.1f}', delta_color = 'inverse')
    m3.metric(label ='Niveau de Risque :',value = f'{st.session_state.proba*100:.1f} %', delta = 'probabilité de defaut', delta_color = 'inverse')
    fig,ax=plt.subplots()
    plot_gauge(arrow=st.session_state.proba, threshold=st.session_state.threshold100/100, n_colors=50,title= 'risque', ax=ax)
    m4.write(fig)


# --------------------------------------------------------
# Explain client data
@st.cache()
def get_client_explain(id, threshold=None, return_data=False)->Union[ClientExplainResponse, ErrorResponse]:
    """explain give loan or not"""
    params=dict(return_data=return_data)
    if not threshold is None:
        params['threshold']= threshold
    response = requests.get(f'{API_URL}/explain/{id}', params=params)
    data = response.json()
    if data.get('error'):
        st.write(data)
    return data



def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def st_shap2(plot, height=None):
    fig= plt.gcf() if plot is None else plot
    st.write(fig)
 

def show_local_explain(col):
    # Explain
    explain_data = get_client_explain(st.session_state.client_id, st.session_state.threshold, return_data=True)
    if not explain_data.get('error'):
        shap_values=series_from_dictkey(explain_data,'shap_values').to_numpy()
        expected_value=explain_data.get('expected_value')
        client_data=series_from_dictkey(explain_data,'client_data')
        feature_names=client_data.index.tolist()
        client_values= client_data.to_numpy()
        st.session_state.client_data=client_data
        exp = shap.Explanation(shap_values,
                base_values=expected_value,
                feature_names=feature_names,
                data=client_values
                    )
        with col:
            d1='<span style="color:#FFF;background-color:#FF0051; padding:5px;margin: 10px;">augmentation de risque</span>'
            d2='<span style="color:#FFF;background-color:#008BFB; padding:5px;margin: 10px;">reduction de risque</span>'
            v1,v2=st.columns(2)
            v1.write(d1,unsafe_allow_html=True)
            v2.write(d2,unsafe_allow_html=True)

            # Interprétabilité locale
            st.write('Waterfall plot')
            plt.close()
            st_shap2(shap.plots.waterfall(exp, max_display=10))
            st.write('Force plot')
            plt.close()
            st_shap(shap.plots.force(exp))



def main():
    """Display data"""
    initialise_session_state()
    show_header()
    a1, a2= st.columns((1,1))
    show_select_client(a1)
    show_threshold_slider(a2)
    update_client_data()
    mc=st.container()    
    show_metrics(mc)
            
    expander= st.expander('Variables les plus influentes pour ce client',expanded=True)
    show_local_explain(expander)

    df_client =st.session_state.client_data.to_frame()
    if isinstance(df_client, pd.DataFrame):
        st.dataframe(df_client)


# ------------------------------------------------
# Utility functions
# ------------------------------------------------
def series_from_dictkey(data,key:str)->pd.Series:
    """
    Convert a dictionary held within a dictionary key to a series.

    Most response json are dictionaries within dictionaries (hierarchical json).
    Use this method to extract a series
    returns empty series otherwise
    """
    dict_key=data.get(key,{})
    nb_keys= len(dict_key.keys())
    df = pd.DataFrame.from_dict(data.get(key,{}),orient='index')
    #st.write(f'series_from_dictkey(data,key={key}, nb = {nb_keys}, df.shape={df.shape}')
    return df.iloc[:,0]


# ------------------------------------------------
# Requests to API server
# ------------------------------------------------


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