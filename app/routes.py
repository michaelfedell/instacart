from flask import render_template, request, redirect, url_for, logging
from app import app, db
from app.models import OrderType
import random
import json

import logging


logger = logging.getLogger(__name__)


@app.route('/')
def index():
    return 'hello_world'


@app.route('/result')
def prediction():
    request.args.get('')

