# To generate a new secret key:
# import random, string
# SECRET_KEY= "".join([random.choice(string.printable) for _ in range(24)])
# print (SECRET_KEY)

SECRET_KEY = "f8p|Ij0c{_P/r|{nb>Y/FmSE"

MODEL_SERVER ='./saved_models'
MODEL_FILE='lgbm_best_model.pickle'
DATA_SERVER = './saved_models'
DATA_FILE='x_test.pickle'
