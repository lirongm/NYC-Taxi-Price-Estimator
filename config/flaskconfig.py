import os

DEBUG = True
LOGGING_CONFIG = "config/logging/logging.conf"
PORT = 5000
APP_NAME = "NYC_Taxi_Fare"
#SQLALCHEMY_D
SQLALCHEMY_TRACK_MODIFICATIONS = True
HOST = "0.0.0.0"
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed

# Connection string
MYSQL_USER = os.environ.get('MYSQL_USER')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD')
MYSQL_HOST = os.environ.get('MYSQL_HOST')
MYSQL_PORT = os.environ.get('MYSQL_PORT')
DATABASE_NAME = os.environ.get('DATABASE_NAME')
MYSQL_DIALECT = 'mysql+pymysql'
SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')

# If host is not specified, create a database locally
if SQLALCHEMY_DATABASE_URI is not None:
    pass
elif MYSQL_HOST is None:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///data/prediction.db'
else:
    SQLALCHEMY_DATABASE_URI = '{dialect}://{user}:{pw}@{host}:{port}/{db}'.format(dialect=MYSQL_DIALECT, user=MYSQL_USER,
                                                                                  pw=MYSQL_PASSWORD, host=MYSQL_HOST, port=MYSQL_PORT,
                                                                                  db=DATABASE_NAME)


# delay between geocoding calls
MIN_DEALY_SECONDS = 1

# trained model path
MODEL_PATH = "model/model.pkl"

# dictionary to indicate what features need to be one-hot encoded
# list all possible values for each feature for one-hot encoding
ONE_HOT_ENCODER = {
    'pickup_dayofweek': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    'pickup_hour': [str(x) for x in range(24)]}
