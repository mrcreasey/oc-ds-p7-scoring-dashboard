from flask import Flask, jsonify, request
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import pickle
import shap

from api.models import ClientPredictResponse, ErrorResponse

# Intitialisation
# to run locally:
# 
app = Flask(__name__)
# Config options - Make sure you created a 'config.py' file.
app.config.from_object('api.config')
# To get one variable, tap app.config['MY_VARIABLE']

model_server=app.config['MODEL_SERVER']
model_file=app.config['MODEL_FILE']
data_server=app.config['DATA_SERVER']
data_file=app.config['DATA_FILE']
explainer_file=app.config['EXPLAINER_FILE']
default_threshold=app.config['THRESHOLD']

model_path=f'{model_server}/{model_file}'
explainer_path=f'{model_server}/{explainer_file}'
data_path=f'{data_server}/{data_file}'

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

Model = imbpipeline or Pipeline or LGBMClassifier
Explainer = shap.TreeExplainer or shap.LinearExplainer

model: Model  = load_pickle(model_path)
explainer: Explainer = load_pickle(explainer_path)

# Load client data
data:pd.DataFrame = load_pickle(data_path)
max_records= 500
if len(data)>max_records:
    data=data.head(max_records)

# data.index should already have been set to SK_ID_CURR
# this is so we do not have to drop the column before making predictions
if 'SK_ID_CURR' in data.columns:
    data=data.set_index('SK_ID_CURR')
list_clients=list(data.index)


@app.route('/')
@app.route('/index/')
def index():
    """list available api routes"""
    routes = ['/clients/', '/client/{id}', '/predict/{id}', '/explain/{id}']

    htmlstr= '<html style="font-family: Roboto, Tahoma, sans-serif;"><body>'
    htmlstr+= '<p>valid routes are :<p><ul>'
    for route in routes:
        htmlstr+=f'<li>{route}</li>'
    htmlstr+='</ul></body></html>'    
    response = htmlstr
    return response 

@app.route('/clients/',  methods=['GET'])
def clients():
    """return (test) list of clients"""
    # response = data[['SK_ID_CURR']].to_json()
    response = jsonify(list_clients)
    return response

def get_client_data(df:pd.DataFrame, id:int) :
    """Renvoie les données d'un client """
    # client_data= data[data['SK_ID_CURR']==int(id)]
    client_data = df[df.index==int(id)]
    if len (client_data) > 0:
        return client_data

@app.route('/client/<id>',  methods=['GET'])
def client(id:int):
    """Renvoie les données d'un client """
    client_data = get_client_data(data,id)
    if client_data is None:
        response = ErrorResponse(error="Client inconnu")
    else:
        response = client_data.iloc[0].to_dict()
    return jsonify(response)

def is_true(ch:str or bool)-> bool:
    if isinstance(ch,bool):
        return ch==True
    if isinstance(ch,str):
        return ch.lower() in ['true', '1', 't', 'y']
    return False

@app.route('/predict/<id>',  methods=['GET'])
def predict(id:int):
    """
    Renvoie le score d'un client en réalisant 
    le predict à partir du modèle final sauvegardé
    Example :
    - http://127.0.0.1:5000/predict/395445?threshold=0.3&return_data=y
    """
    client_data = get_client_data(data,id)
    if client_data is None:
        response = ErrorResponse(error="Client inconnu")
    else:
        threshold = request.args.get('threshold',default_threshold)
        if isinstance(threshold,str):
            threshold = float(threshold)
        return_data= is_true(request.args.get('return_data',False))  # type: ignore
        y_pred_proba = model.predict_proba(client_data)[:,1]
        y_pred_proba = y_pred_proba[0]
        y_pred= int((y_pred_proba > threshold)*1)
        client_data=client_data.iloc[0].to_dict() if return_data else {}
        response = ClientPredictResponse(
            id=id,
            y_pred_proba= y_pred_proba,
            y_pred= y_pred,
            model_type=f'{type(model)}',
            client_data=client_data
        )
    return jsonify(response)


@app.route('/explain/<id>',  methods=['GET'])
def explain(id:int):
    """
    Renvoie le score d'un client en réalisant 
    le predict à partir du modèle final sauvegardé
    Example :
    - http://127.0.0.1:5000/predict/395445?threshold=0.3&return_data=y
    """
    client_data = get_client_data(data,id)
    if client_data is None:
        response=jsonify(error="Client inconnu")
    else:
        threshold = request.args.get('threshold',default_threshold)
        if isinstance(threshold,str):
            threshold=float(threshold)
        return_data= is_true(request.args.get('return_data',False))  # type: ignore
        y_pred_proba = model.predict_proba(client_data)[:,1]
        y_pred_proba=y_pred_proba[0]
        y_pred= int((y_pred_proba > threshold)*1)
        # get first data row (as series)
        feature_names=list(client_data.columns)
        client_data=client_data.head(1)
        explainer_shap_values = explainer.shap_values(client_data, check_additivity=False)
        shap_values = pd.Series(data=explainer_shap_values[0], index=feature_names).to_dict() 
        expected_values = pd.Series(data=explainer.expected_value, index=feature_names).to_dict() 
        client_data= client_data.iloc[0].to_dict() if return_data else {}
        response = jsonify(
            id=id,
            y_pred_proba= y_pred_proba,
            y_pred= y_pred,
            shap_values=shap_values,
            expected_values= expected_values,
            client_data=client_data
        )
    return response


# python api/app.py -> runs locally on localhost port 5000
# python run.py -> correct path to imports (as on heroku)
if __name__ == "__main__":
    app.run(debug=False)