from flask import render_template, request, redirect, url_for, logging
from app import app, db
from app.models import OrderType
import random
import json
import os
import pandas as pd

from src.helpers import get_newest_model, get_files
from src.score_model import expand_factors, map_user_input, predict
import logging


logger = logging.getLogger(__name__)

tmo = get_newest_model(get_files('models', file_filter='*.pkl'))
factors = pd.read_csv('data/features/factors.csv')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result')
def prediction():
    # These options are on same scale as features and just need to be taken from string
    options = ['veg', 'gf', 'xlac', 'jon', 'frequency']
    factor_data = {opt: eval(request.args.get(opt, 'False')) for opt in options}
    # Sliders are on -300:300 range for smooth sliding - need to scale down
    sliders = ['habit', 'health', 'time']
    for s in sliders:
        factor_data[s] = int(request.args.get(s, 0)) / 100
    app.logger.debug('Parsed user input: %s', factor_data)
    x = map_user_input(factors, **factor_data, cols=factors.columns)
    prediction = tmo.predict(x)
    order_type = OrderType.query.filter_by(label=int(prediction)).first()
    app.logger.info('Prediction: %d; OrderType: %s', prediction, order_type)
    return render_template('result.html', order_type=order_type)

