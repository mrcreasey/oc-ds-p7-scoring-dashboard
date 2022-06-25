# Flask API deployment

This folder ('/api/') contains the Flask API and a configuration file

To deploy to Heroku:
- ensure gunicorn is in root folder `requirements.txt` (or another web server)
- ensure flask and other required libraries are in `requirements.txt`
- ensure all files used by this application are included in git (notably saved_models)
- In the root folder:
  - `Procfile` contains initialisation instruction `web: gunicorn api.app:app`
  - `runtime.txt` informs Heroku which python version to use (tested using python-3.7.13)
  - `run.py` is currently only used for local development 

Follow instructions at <https://devcenter.heroku.com/> :
- install Heroku CLI
- In terminal:
  - `heroku login` (you will be asked to login)
  - `heroku create [app-name] --region eu`  (if no app-name, heroku generates a name, if no region heroku assigns region us)
  - `heroku run init`
  
After any modifications to the flask app code or configuration, git commit then run the following command :  
  - `git push heroku main` (or name of your branch)

Your app will be built and deployed to `https://app-name.herokuapp.com/`

If app fails to build, type `heroku logs` to identify reason for failure
