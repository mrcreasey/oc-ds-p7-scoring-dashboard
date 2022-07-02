# Flask API

This folder ('/api/') contains the Flask API (with a configuration file `config.py`)

## API calls

(Currently using custom api endpoints)

| HTTP method | API endpoint                            | Description                                              |
| ----------- | --------------------------------------- | -------------------------------------------------------- |
| GET         | clients                                 | Get a list of clients                                    |
| GET         | clients/<id>                            | Get a single client                                      |
| GET         | clients/<id>?predict=True               | Get a single client and predict risk (default threshold) |
| GET         | clients/<id>?predict=True&threshold=0.7 | Get a single client and predict risk (custom threshold)  |
| GET         | clients/<id>?explain=True               | Get a single client and explain risk (default threshold) |

## To deploy to Heroku:

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
  - `heroku create [app-name] --region eu` (if no app-name, heroku generates a name, if no region
    heroku assigns region us)
  - `heroku run init`

After any modifications to the flask app code or configuration, git commit then run the following
command :

- `git push heroku main` (or name of your branch)

Your app will be built and deployed to `https://app-name.herokuapp.com/`

If app fails to build, type `heroku logs` to identify reason for failure

## Reading data from another source

For Proof of Concept (POC), this application reads the data locally on Heroku.

For production, the code should be modified to read from an external source (with authentification, encryption and add a library for in-memory cache):

For example, read from a bucket S3 on AWS, obtaining access through ‘secret’ config vars :
```bash
$ heroku login
$ heroku config:set S3_KEY=8N029N81 S3_SECRET=9s83109d3+583493190 S3_BUCKET=MC-OC-7
```

Then in the source code `api/app.py`:
```python 
import s3fs
S3_BUCKET = os.environ['S3_BUCKET']
S3_KEY = os.environ['S3_KEY']; S3_SECRET = os.environ['S3_SECRET'])
fs = s3fs.S3FileSystem(anon=False, access_key=S3_KEY, secret_key=S3_SECRET)

def read(filename)
    with fs.open(f's3://{S3_BUCKET}/{filename}') as f:
        return pd.read_csv(f)

data = read('x_test.csv')        
```

See <https://s3fs.readthedocs.io/en/latest/> for examples
