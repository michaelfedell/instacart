import os
import logging

logger = logging.getLogger(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))


DATABASE_NAME = 'instacart'
SQLITE_DB_STRING = 'sqlite:///' + os.path.join(basedir, 'data/{}.db'.format(DATABASE_NAME))
SECRET_KEY = os.environ.get('SECRET_KEY') or 'default-session-key'
conn_type = 'mysql+pymysql'
user = os.environ.get('MYSQL_USER')
password = os.environ.get('MYSQL_PASSWORD')
host = os.environ.get('MYSQL_HOST')
port = os.environ.get('MYSQL_PORT')
RDS_DB_STRING = '{}://{}:{}@{}:{}/{}'\
                .format(conn_type, user, password, host, port, DATABASE_NAME)
LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT')

DEBUG = True
LOGGING_CONFIG = 'config/logging/local.conf'
PORT = 5000
APP_NAME = 'instacart'
mode = os.environ.get('MODE', 'local')
SQLALCHEMY_DATABASE_URI = SQLITE_DB_STRING if mode == 'local' else RDS_DB_STRING
SQLALCHEMY_TRACK_MODIFICATIONS = True
HOST = '127.0.0.1'
SQLALCHEMY_ECHO = False  # If true, SQL for queries made will be printed
MAX_ROWS_SHOW = 100
