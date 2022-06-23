from flask import Flask, jsonify, request
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
import pickle

# Intitialisation
app = Flask(__name__)
# Config options - Make sure you created a 'config.py' file.
app.config.from_object('config')
# To get one variable, tape app.config['MY_VARIABLE']

model_server=app.config['MODEL_SERVER']
model_file=app.config['MODEL_FILE']
data_server=app.config['DATA_SERVER']
data_file=app.config['DATA_FILE']
default_threshold=app.config['THRESHOLD']

model_path=f'{model_server}/{model_file}'
data_path=f'{data_server}/{data_file}'

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

model: LGBMClassifier = load_pickle(model_path)


data= pd.DataFrame({
    'SK_ID_CURR':[567,123,875],
    'firstname':['Jean','Karen','Bob'],
    'lastname':['Martin','Moreau','Holly'],
    })

# Load client data
data:pd.DataFrame = load_pickle(data_path)
if len(data)>50:
    data=data.head(50)

# data.index should already have been set to SK_ID_CURR
# this is so we do not have to drop the column before making predictions
if 'SK_ID_CURR' in data.columns:
    data=data.set_index('SK_ID_CURR')
list_clients=list(data.index)


@app.route('/')
@app.route('/index/')
def index():
    """list available api routes"""
    routes = ['/clients/', '/client/{id}', '/predict/{id}']

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

def get_client_data(df:pd.DataFrame,id) :
    """Renvoie les données d'un client """
    # client_data= data[data['SK_ID_CURR']==int(id)]
    client_data= df[df.index==int(id)]
    if len (client_data) > 0:
        return client_data

@app.route('/client/<id>',  methods=['GET'])
def client(id):
    """Renvoie les données d'un client """
    client_data = get_client_data(data,id)
    if client_data is None:
        response=jsonify(error="Client inconnu")
    else:
        clien=client_data.iloc[0].to_dict()
        response= jsonify(clien)
        # response = client_data.iloc[0].to_json()
    return response

def is_true(ch:str or bool)-> bool:
    if isinstance(ch,bool):
        return ch==True
    if isinstance(ch,str):
        return ch.lower() in ['true', '1', 't', 'y']
    return False

@app.route('/predict/<id>',  methods=['GET'])
def predict(id):
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
        client_data=client_data.iloc[0].to_dict() if return_data else {}
        response = jsonify(
            id=id,
            y_pred_proba= y_pred_proba,
            y_pred= y_pred,
            client_data=client_data
        )
    return response


# python api/app.py -> runs locally on localhost port 5000
if __name__ == "__main__":
    app.run(debug=True)