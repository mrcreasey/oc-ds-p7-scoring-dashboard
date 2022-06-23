# Streamlit dashboard deployment

This folder ('/dashboard/') contains the streamlit dashboard 

Streamlit is configured through configuration files in a hidden root folder `.streamlit` (gitignore) :
- `config.toml` : contains theming, debugging etc
- `secrets.toml` : contains secret keys, flask api url etc.

To deploy to streamlit :`
- create an account at <https://streamlit.io/> linked to your github account
- authorise github access
- sign in at <https://share.streamlit.io/>
- deploy by clicking `new app` (choose repository, branch, file to deploy = `dashboard/main.py`)
- in `app settings, secrets`, if `.streamlit` folder is not saved to git:
  - copy-paste theme from file `.streamlit/config.toml`
  - copy-paste secrets from file `.streamlit/secrets.toml`