from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world !"

# python api/app.py -> runs locally on localhost port 5000
if __name__ == "__main__":
    app.run()