import glob
import logging
import os
import pickle

from sklearn.base import ClassifierMixin


logger = logging.getLogger(__name__)


def get_files(dir_path, file_filter=None, recursive=False):
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

