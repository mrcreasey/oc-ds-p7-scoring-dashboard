from flask import Flask
import pandas as pd

app = Flask(__name__)

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
    response = pd.DataFrame([567,123,875], columns=['SK_ID_CURR']).to_json()
    return response

# python api/app.py -> runs locally on localhost port 5000
if __name__ == "__main__":
    app.run()