import argparse
import datetime as dt
import logging
import os
import pickle

import pandas as pd
import yaml
import boto3

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression as logistic
from sklearn.ensemble import RandomForestClassifier as random_forest
from sklearn.ensemble import GradientBoostingClassifier as gradient_boosting
from sklearn.svm import LinearSVC as linear_svc
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
now = dt.datetime.now().replace(microsecond=0).isoformat()


def get_data(data_path, config):
    """
    Load the feature data and split for train/test and X/Y

    Args:
        data_path (str): path to the data file with features (shoppers.csv)
        config (dict): dictionary of configuration from model_config.yml

    Returns:
        dict: Dictionary of data after being split into parts
            x_train: training set (features only)
            x_test: test set (features only)
            y_train: targets for training set (same order as X)
            y_test: targets for test set (same order as X)
            train_weights: weights to balance classes among training data

    """
    logger.info('Reading data from %s', data_path)
    users = pd.read_csv(data_path, dtype={'label': int}).set_index('user_id')
    logger.debug('%d observations loaded', len(users))
    X = users.drop(columns='label')
    if config.get('scale_data'):
        logger.debug('Scaling data to standard normal')
        X = preprocessing.scale(X)
    Y = users['label']
    logger.debug('%0.1f%% of data held for testing', 100*config['split']['test_size'])
    data = train_test_split(X, Y, **config['split'])
    x_train, x_test, y_train, y_test = data
    train_weights = class_weight.compute_sample_weight('balanced', y_train)

    return {'x_train': x_train, 'x_test': x_test,
            'y_train': y_train, 'y_test': y_test,
            'train_weights': train_weights}


def save_split(data):
    """
    Save the individual parts of data after splitting

    Args:
        data (dict): data dictionary with keys x_train, etc. produced by get_data()

    Returns:

    """
    logger.debug('Saving split features for posterity')
    for part in data:
        if len(data[part]) == 0:
            logger.debug('%s contains 0 rows, skipping filesave', part)
            continue
        path = f'features/{part}-features_{now}.csv'
        pd.DataFrame(data[part]).to_csv(path)
        logger.info('Saved %s to %s', part, path)


def train_model(x_train, y_train, config, sample_weight=None):
    """
    Fit the config-specified model to training data

    Args:
        x_train (np.ndarray): training data (features only)
        y_train (np.ndarray): targets for training data (same order as x)
        sample_weight (optional, np.ndarray): defaults to None; weights for each sample to balance dataset
                                              (if passed, must have same len and order as x_train)
        config (dict): dictionary of configurations from model_config.yml

    Returns:
        trained model object of (scikit-learn model) class specified by config

    """
    model_type = config.get('model')  # str matching imported model aliases
    model = eval(model_type)  # actual class object imported from sklearn
    logger.debug('Training %s model (%s)', model_type, model)
    tmo = model(**config.get(model_type))  # instantiated model class with config
    logger.debug('Fitting model to %d training observations', len(x_train))
    tmo.fit(x_train, y_train, sample_weight)  # model object fit to training data
    logger.debug('Model training completed')
    return tmo


def evaluate_model(model, x_test, y_test):
    """
    Evaluate model against test set to get idea of performance

    Args:
        model (tmo): Trained model object ready to predict
        x_test (np.ndarray): test data (features only)
        y_test (np.ndarray): targets for test data (same order as x)

    Returns:
        (float, float): accuracy, f1_score

    """
    if not len(x_test) > 0:
        logger.warning('Tried to evaluate model with empty test set')
        return None
    pred = model.predict(x_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=pred, normalize=True)
    f1 = f1_score(y_true=y_test, y_pred=pred, average='weighted')
    logger.info('Model predicts with %0.1f%% accuracy and F-1 score of %0.1f%%',
                accuracy * 100, f1 * 100)
    return accuracy, f1


def save_model(model, output_path):
    """
    Save the trained model to a local directory

    Args:
        model (tmo): trained model object (pickle-serializable)
        output_path (str): path to save model (will generate filename if directory passed)

    Returns:
        str: actual path to saved model (accounts for any name corrections made)

    """
    # If directory is given (with no file name), filename generated with datestamp
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, 'tmo_{}.pkl'.format(now))
        logger.debug('Provided output path is a directory, defaulting filename to %s', output_path)
    # Will add a version number to avoid overwrite;
    # version number incremented until unique filename produced
    base, ext = os.path.splitext(output_path)
    version = 1
    while os.path.exists(output_path):
        version += 1
        output_path = f'{base}_v{version}{ext}'
    with open(output_path, 'wb') as f:
        pickle.dump(model, f, -1)
    logger.info('Trained model object saved to %s', output_path)
    logger.info('Setting env var TMO_PATH to find this most-recent model')
    os.environ['TMO_PATH'] = output_path
    return output_path


def upload_model(model_path, bucket_name):
    """
    Upload saved model object to S3.

    Args:
        model_path: path to pickled tmo (also used as key in s3
        bucket_name: name of s3 bucket for use (need access)

    Returns:

    """
    s3 = boto3.client('s3')
    logger.debug('Uploading model to %s bucket with key: %s', bucket_name, model_path)
    s3.upload_file(model_path, bucket_name, model_path)


def run_train_model(args):
    """Call model training steps to run script"""
    with open(args.config) as f:
        config = yaml.load(f)
    data = get_data(os.path.abspath(args.data), config)
    sample_weights = data['train_weights'] if config.get('weight_classes') else None
    model = train_model(data['x_train'], data['y_train'], config, sample_weights)
    if len(data['x_test']) > 0:
        accuracy, f1 = evaluate_model(model, data['x_test'], data['y_test'])
    if args.save_split:
        save_split(data)
    if args.output:
        model_path = save_model(model, args.output)
        if args.upload:
            upload_model(model_path, args.bucket)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model on feature-rich data")
    parser.add_argument('-c', '--config', default='config/model_config.yml',
                        help='path to yaml file with model configurations')
    parser.add_argument('-d', '--data', default='data/features/shoppers.csv',
                        help='Path to shopper data file (may be local or s3)')
    parser.add_argument('-o', '--output', default='models/',
                        help='path to yaml file with model configurations')
    parser.add_argument('-u', '--upload', action='store_true',
                        help='If true will upload trained model object to s3 bucket')
    parser.add_argument('-b', '--bucket', default='instacart-store',
                        help='Name of S3 bucket to upload files')
    parser.add_argument('--save_split', default=None,
                        help='Prefix to save split feature files for posterity. None to skip this step.')

    args = parser.parse_args()

    run_train_model(args)
