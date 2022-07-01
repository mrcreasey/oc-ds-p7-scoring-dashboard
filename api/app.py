from pyexpat import features
from flask import Flask, jsonify, request
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.linear_model import LogisticRegression
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
data:pd.DataFrame=pd.DataFrame()

# Load client data 
if data_file.endswith('pickle'):
    # preprocessed data for Proof of Concept
    data:pd.DataFrame = load_pickle(data_path)
else:
    # Raw data files
    # In production, conduct an authenticated, authorised read CSV from AWS S3 bucket
    data:pd.DataFrame = pd.read_csv(data_path)

max_records = 1000 # Proof of concept (POC): limité pour accélerer le temps de réponse
if len(data)>max_records:
    data=data.head(max_records)

# data.index should already have been set to SK_ID_CURR
# this is so we do not have to drop the column before making predictions
if 'SK_ID_CURR' in data.columns:
    data=data.set_index('SK_ID_CURR')

list_clients=list(data.index)

# Par défaut, le classifier est le model, et les données n'ont pas besoin de preprocess
data_prep=data
clf = model
type_model=type(model)

if isinstance(model, imbpipeline) or isinstance(model,Pipeline):
    # shap n'est pas capable de travailler sur les pipelines
    # il faut extraire le classificateur et preprocess les données (si besoin)
    clf=model.named_steps['clf']
    type_model= type(clf)
    # on enleve le classificateur pour faire le preprocessing/feature_selection des données
    data_prep=pd.DataFrame(model[:-1].transform(data),index=data.index, columns=data.columns)


print(f'init_app, clf = {type_model}')
# if explainer is None:
#     if isinstance(clf,LGBMClassifier):
#         explainer=shap.TreeExplainer(clf,data_prep)
#     elif isinstance(clf,LogisticRegression):
#         explainer= shap.LinearExplainer(clf,data_prep)    

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
    global type_model
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
            model_type=f'{type_model}',
            client_data=client_data
        )
    return jsonify(response)


def df_to_json(df:pd.DataFrame)->dict:
    """convert dataframe to json
    https://stackoverflow.com/a/26646362/
    """
    return dict(
        feature_names=list(df.columns),
        # dtypes=list(df.dtypes),
        shape=df.shape,
        data=df.to_numpy().tolist()
    )

def json_to_df(jd:dict)->pd.DataFrame:
    """convert json to dataframe
    https://stackoverflow.com/a/26646362/
    """
    df= pd.DataFrame(np.array(jd.get('data')), columns=jd.get('feature_names'))
    # print (f'json_to_df, {jd.get("shape")}')
    # print (f'json_to_df, {df.shape}')
    return df


@app.route('/explain/',  methods=['GET'])
def explain_all():
    """
    Renvoie les explications shap de jusqu'à 1000 clients à partir du modèle final sauvegardé
    Utilisé pour afficher les beeswarm et summary plots
    Example :
    - http://127.0.0.1:5000/explain/nb=100
    """ 
    global data, model, explainer
    sample_size:int = int(request.args.get('nb',100))
    max_sample_size=1000
    nb= min(max_sample_size,sample_size,len(data))
    data_sample:pd.DataFrame=data.sample(n=nb, random_state=42)
    # preprocess
    data_sample_prep = data_sample
    if isinstance(model,imbpipeline) or isinstance(model,Pipeline):
        data_sample_prep = pd.DataFrame(model[:-1].transform(data_sample), 
                index=data_sample.index,columns=data_sample.columns)
    client_data_json= df_to_json(data_sample_prep)
    shap_values = explainer.shap_values(data_sample_prep, check_additivity=False).tolist()
    expected_value = explainer.expected_value  # only keep class 1)
    response = jsonify(
        shap_values=shap_values,
        expected_value= expected_value,
        client_data=client_data_json
    )
    return response


@app.route('/explain/<id>',  methods=['GET'])
def explain(id:int):
    """
    Renvoie les explications shap d'un client à partir du modèle final sauvegardé
    Example :
    - http://127.0.0.1:5000/explain/395445?threshold=0.3&return_data=y
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
        # preprocess
        client_data_prep = client_data
        if isinstance(model,imbpipeline) or isinstance(model,Pipeline):
            client_data_prep = pd.DataFrame(model[:-1].transform(client_data), 
            index=client_data.index,columns=client_data.columns)
        explainer_shap_values = explainer.shap_values(client_data_prep, check_additivity=False)
        shap_values = pd.Series(data=explainer_shap_values[0], index=feature_names).to_dict() 
        expected_value = explainer.expected_value 
        client_data= client_data.iloc[0].to_dict() if return_data else {}
        response = jsonify(
            id=id,
            y_pred_proba= y_pred_proba,
            y_pred= y_pred,
            shap_values=shap_values,
            expected_value= expected_value,
            client_data=client_data
        )
    return response


# python api/app.py -> runs locally on localhost port 5000
# python run.py -> correct path to imports (as on heroku)
if __name__ == "__main__":
    app.run(debug=False)