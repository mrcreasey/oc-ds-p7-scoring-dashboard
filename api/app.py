from flask import Flask, jsonify
import numpy as np
import pandas as pd

# Intitialisation
app = Flask(__name__)

data= pd.DataFrame({
    'SK_ID_CURR':[567,123,875],
    'firstname':['Jean','Karen','Bob'],
    'lastname':['Martin','Moreau','Holly'],
    })


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
    response = data[['SK_ID_CURR']].to_json()
    return response

@app.route('/client/<id>',  methods=['GET'])
def client(id):
    """Renvoie les données d'un client """
    client_data= data[data['SK_ID_CURR']==int(id)]
    response = client_data.to_json()
    return response


@app.route('/predict/<id>',  methods=['GET'])
def predict(id):
    """
    Renvoie le score d'un client en réalisant 
    le predict à partir du modèle final sauvegardé
    """
    client_data= data[data['SK_ID_CURR']==int(id)]
    if len (client_data) > 0:
        response = jsonify({"id":id,"y_pred":np.random.random()})
    else:
        response={"error_msg":"Client inconnu"}
    return response


# python api/app.py -> runs locally on localhost port 5000
if __name__ == "__main__":
    app.run(debug=True)