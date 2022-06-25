# Streamlit dashboard deployment

This folder ('/dashboard/') contains the streamlit dashboard 

Streamlit is configured through configuration files in a hidden root folder `.streamlit` (gitignore) :
- `config.toml` : contains theming, debugging etc
- `secrets.toml` : contains secret keys, flask api url etc.

## To deploy locally
- Run command `streamlit run dashboard/main.py`

## To deploy to streamlit
- create an account at <https://streamlit.io/> linked to your github account
- authorise github access
- sign in at <https://share.streamlit.io/>
- deploy by clicking `new app` (choose repository, branch, file to deploy = `dashboard/main.py`)

## Secrets Management

- <https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management>
- in `app settings, secrets`, if `.streamlit` folder or following files are not saved to git:
  - copy-paste theme from file `.streamlit/config.toml`
  - copy-paste secrets from file `.streamlit/secrets.toml`
