import yaml
from flask import render_template, request, redirect, url_for, flash
from app import app, db
from app.models import OrderType
import random
import json
import os
import pandas as pd

from src.helpers import get_newest_model, get_files
from src.score_model import expand_factors, map_user_input, predict, predict_file
import logging


logger = logging.getLogger(__name__)
with open('config/features_config.yml') as f:
    fc = yaml.load(f)

tmo = get_newest_model(get_files('models', file_filter='*.pkl'))
if not tmo:
    tmo = get_newest_model(get_files('s3://{}/models'.format(fc.get('s3_bucket-name')),
                                     file_filter='*.pkl'))
factors = pd.read_csv('data/features/factors.csv')
features = factors.columns
days = {i: d for i, d in enumerate(['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                                    'Thursday', 'Friday', 'Saturday'])}
clust_type = fc.get('cluster-method')
n_clust = fc.get('gmm').get('n_components') if clust_type == 'gmm' else fc.get('kmeans').get('k')

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
    x = map_user_input(factors, **factor_data, cols=features)
    label = predict(tmo, x)
    order_type = OrderType.query.filter_by(label=int(label)).first()
    app.logger.info('Prediction: %d; OrderType: %s', label, order_type)
    return render_template('result.html', order_type=order_type.__dict__,
                           days=days)


@app.route('/upload', methods=['GET', 'POST'])
def profiles():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            logger.warning('No file part')
            return redirect(request.url)
        f = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if f.filename == '':
            logger.warning('No selected file')
            flash('No selected file')
            return redirect('/')

        ftype = os.path.splitext(f.filename)[1]
        if ftype != '.csv':
            logger.error('File must be of type: ".csv", not "{}"'.format(ftype))
            flash('File must be of type: ".csv", not "{}"'.format(ftype))
            return redirect('/')

        try: label, count = predict_file(tmo, f, cols=features)
        except ValueError as e:
            logger.error(e)
            flash(str(e))
            return redirect('/')

        counts = {k: v for k, v in zip(label, count)}
        counts = {i: counts.get(i, 0) for i in range(1, n_clust+1)}
        desc = {i: OrderType.query.filter_by(label=i).first().desc for i in range(1, n_clust+1)}
        return render_template('profiles.html', counts=counts, count=sum(counts.values()), desc=desc)
    else:
        f = 'app/static/shopper_sample.csv'
        try: label, count = predict_file(tmo, f, cols=features)
        except ValueError as e:
            logger.error(e)
            flash(str(e))
            return redirect('/')
        counts = {k: v for k, v in zip(label, count)}
        counts = {i: counts.get(i, 0) for i in range(1, n_clust+1)}
        desc = {i: OrderType.query.filter_by(label=i).first().desc for i in range(1, n_clust+1)}
        return render_template('profiles.html', counts=counts, count=sum(counts.values()), desc=desc)
