#! /usr/bin/env python
from api.app import app
# type 'python run.py' to start the Flask API application
# In local development, the app will be running on http://127.0.0.1:5000
# Note: the Heroku Procfile uses gunicorn web server pointed directly to api.app:app
if __name__ == "__main__":
    app.run(debug=False)
