import argparse
import datetime as dt
import logging
import os
import pickle

import pandas as pd
import yaml

import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as logistic
from sklearn.ensemble import RandomForestClassifier as random_forest
from sklearn.ensemble import GradientBoostingClassifier as gradient_boosting
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVC as linear_svc
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def predict(model, data):
    logger.debug('Predicting labels for %d observations with %d features', data.shape[0], data.shape[1])
    logger.debug('Model is of type %s', model)
    data = data.values if isinstance(data, pd.DataFrame) else data
    if np.any(np.abs(data) > 5):
        logger.warning('Passed data has absolute values greater than 5; prediction'
                       'data should be scaled to standard-normal')
    return model.predict(data)


def expand_factors(factor_components, factor_data):
    """

    Args:
        factor_components (np.ndarray or pd.DataFrame): Factor components matrix
        factor_data (np.ndarray or pd.DataFrame): Observations represented by low-dimension factors

    Returns:
        np.ndarray: Observations represented in full-dimensional feature space

    """
    logger.debug('Transforming input from %d x %d to full %d-dimensional feature space',
                 factor_data.shape[0], factor_data.shape[1], factor_components.shape[1])
    return factor_data.dot(factor_components)


def map_user_input(factor_components, habit, frequency, health, time, veg, gf, xlac, jon, cols=None):
    """

    Args:
        factor_components:
        habit:
        frequency:
        health:
        time:
        veg:
        gf:
        xlac:
        jon:
        cols:

    Returns:

    """
    factor_data = np.array([habit, frequency, health, time])
    factor_data = factor_data.reshape(1,4) if (factor_data.ndim == 1) else factor_data
    features = expand_factors(factor_components, factor_data)
    if not isinstance(features, pd.DataFrame):
        if cols is None:
            raise ValueError('Must pass factor_components as pd.DataFrame or pass list of cols')
        features = pd.DataFrame(features, columns=cols)
    for col in ['veg', 'meat', 'fish', 'gluten', 'dairy', 'snack']:
        if col not in features.columns:
            logger.warning('%s does not appear to be in the feature columns;'
                           'skipping all dietary modifications', col)
            return features
    if veg:
        features['veg'] += 2
        features['meat'] -= 2
        features['fish'] -= 2
    if gf:
        features['gluten'] -= 2
    if xlac:
        features['dairy'] -= 2
    if jon:
        features['meat'] += 2
        features['fish'] += 2
        features['snack'] -= 2

    return features


def get_feature_names(factors_path):
    """
    Get feature names by reading header of factors csv

    Args:
        factors_path: path to factors csv with names of features as header

    Returns:
        list(str): list of column names read from file

    """
    with open(factors_path) as f:
        col_names = f.readline().split(',')
        col_names[-1] = col_names[-1].strip('\n')
    # Skip first field if empty (result of  to_csv(save_index=True))
    if not col_names[0]:
        return col_names[1:]
    else:
        return col_names


def predict_file(model, f, cols):
    """

    Args:
        model:
        f:
        cols:

    Returns:

    """
    bad_data = False

    data = pd.read_csv(f)
    if 'user_id' in data.columns:
        data.set_index('user_id', inplace=True)
    if 'label' in data.columns:
        data.drop(columns='label', inplace=True)

    for col in cols:
        if col not in data.columns:
            bad_data = True
            logger.error('Data is missing column for: %s', col)
    if len(data.columns) != 51:
        logger.error('Data does not have the correct number of columns')
        bad_data = True
    x = data.values
    if np.any(np.abs(x) > 5):
        logger.warning('Data has absolute values greater than 5; scaling to standard normal')
        x = scale(x)
    if bad_data:
        raise ValueError('Data was not of the required format')

    labels = model.predict(x)
    return np.unique(labels, return_counts=True)
