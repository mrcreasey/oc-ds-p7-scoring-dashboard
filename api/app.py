from flask import Flask
import pandas as pd

app = Flask(__name__)
data= pd.DataFrame({
    'SK_ID_CURR':[567,123,875],
    'firstname':['Jean','Karen','Bob'],
    'lastname':['Martin','Moreau','Holly'],
    })


@app.route('/')
@app.route('/index/')
def index():
    htmlstr= '<html><p>valid routes are :<p><ul>'
    routes = ['clients', 'client/{id}']
    for route in routes:
        htmlstr+=f'<li>{route}</li>'
    htmlstr+='</ul></html>'    
    response = htmlstr
    return response 

@app.route('/clients/',  methods=['GET'])
def clients():
    # return (test) list of clients
    response = data[['SK_ID_CURR']].to_json()
    return response

@app.route('/client/<id>',  methods=['GET'])
def client(id):
    # return (test) client data
    client_data= data[data['SK_ID_CURR']==int(id)]
    response = client_data.to_json()
    return response

# python api/app.py -> runs locally on localhost port 5000
if __name__ == "__main__":
    app.run(debug=True)