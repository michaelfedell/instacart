import glob
import logging
import os
import re
import pickle
import boto3

from sklearn.base import ClassifierMixin


logger = logging.getLogger(__name__)


def get_files(dir_path, file_filter=None, recursive=False):
    """
    Get all file paths in a subdirectory matching filter expression

    Args:
        dir_path (str): path to subdirectory to search
        file_filter (str): regex filter to apply to search; e.g. "*.csv"
        recursive (bool): perform recursive glob search if true

    Returns:
        list(str): list of file paths resulting from search. Empty list if none found

    """
    if not dir_path:
        logger.warning('Must specify a valid path, not %s', dir_path)

    if not os.path.isdir(dir_path):
        if os.path.exists(dir_path):
            logger.warning('Provided path is not a directory, defaulting to parent dir: %s',
                           dir_path, os.path.pardir(dir_path))
            dir_path = os.path.pardir(dir_path)
        else:
            logger.warning('Path to %s does not exist, returning no files', dir_path)
            return []
    file_filter = file_filter or '*'
    files = glob.glob(os.path.join(dir_path, file_filter), recursive=recursive)
    files = [f for f in files if os.path.isfile(f)]
    return files


def get_newest_model(files):
    """
    Fetch the latest model among the created trained model objects

    Args:
        files (list(str)): list of paths to pickled trained models

    Returns:
        (sklearn Classifier): Trained model object ready to make predictions

    """
    logger.info('Checking %d files for pickled model newest->oldest', len(files))
    for f in sorted(files, key=os.path.getctime, reverse=True):
        try:
            model = pickle.load(open(f, 'rb'))
            logger.info('Returning newest model found at %s', f)
            logger.info('Model is of type %s', model)
            if not isinstance(model, ClassifierMixin):
                logger.warning('Unpickled object was: %s; must be of type sklearn.base.ClassifierMixin', type(model))
                continue
            else:
                return model
        except pickle.UnpicklingError as e:
            logger.error(e)
            logger.error('Could not unpickle %s, moving on to next file', f)
            continue
    logger.warning('No valid model object found. Returning None')
    return None


