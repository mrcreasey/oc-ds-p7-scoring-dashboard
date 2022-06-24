# To generate a new secret key:
# import random, string
# SECRET_KEY= "".join([random.choice(string.printable) for _ in range(24)])
# print (SECRET_KEY)

SECRET_KEY = "f8p|Ij0c{_P/r|{nb>Y/FmSE"

# Location of client data
DATA_SERVER = './saved_models'
DATA_FILE='x_test.pickle'

# Location of saved model
MODEL_SERVER ='./saved_models'
MODEL_FILE='lgbm_best_model.pickle'
EXPLAINER_FILE='lgbm_explainer.pickle'

# best threshold of saved model
THRESHOLD=0.542