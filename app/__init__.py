from flask import Flask
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
import logging.config


# Initialize the Flask application
app = Flask(__name__)
app.config.from_pyfile('../config.py')

# Define LOGGING_CONFIG in flask_config.py - path to config file for setting
# up the logger (e.g. config/logging/local.conf)
logging.config.fileConfig(app.config["LOGGING_CONFIG"])

# Add booststrap styles/framework
bootstrap = Bootstrap(app)

# Initialize the database
db = SQLAlchemy(app)

from app import routes, models
