from flask import Flask, jsonify
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

model_path=f'{model_server}/{model_file}'
data_path=f'{data_server}/{data_file}'

def load_pickle(filename):
    with open(filename, 'rb') as handle:
        return pickle.load(handle)

model= load_pickle(model_path)


data= pd.DataFrame({
    'SK_ID_CURR':[567,123,875],
    'firstname':['Jean','Karen','Bob'],
    'lastname':['Martin','Moreau','Holly'],
    })

# SK_    
data:pd.DataFrame = load_pickle(data_path)
print(data.shape)
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
        return client_data.iloc[0]

@app.route('/client/<id>',  methods=['GET'])
def client(id):
    """Renvoie les données d'un client """
    client_data = get_client_data(data,id)
    if client_data is None:
        response=jsonify(error="Client inconnu")
    else:
        response = client_data.to_json()
    return response


@app.route('/predict/<id>',  methods=['GET'])
def predict(id):
    """
    Renvoie le score d'un client en réalisant 
    le predict à partir du modèle final sauvegardé
    """
    # client_data= data[data['SK_ID_CURR']==int(id)]
    client_data= data[data.index==int(id)]
    if len (client_data) > 0:
        response = jsonify({"id":id,"y_pred":np.random.random()})
    else:
        response={"error_msg":"Client inconnu"}
    return response


# python api/app.py -> runs locally on localhost port 5000
if __name__ == "__main__":
    app.run(debug=True)