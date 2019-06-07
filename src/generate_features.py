import logging
import os

import boto3
import numpy as np
import pandas as pd
import yaml
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score as score_ch
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import FactorAnalysis

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# File should be run from top level dir in project via `make features`
ROOT = os.getcwd()
logger.debug('Project Root = %s', ROOT)


def load_data(data_dir, file_name):
    path = os.path.join(data_dir, file_name)
    logger.debug('Reading data from %s', path)
    return pd.read_csv(path)


def clean(X):
    nans = np.isnan(X).any(axis=1)
    logger.debug('Removing %d rows with nan values in order features',
                 sum(nans))
    return X[~nans]


def gmm_clust(X, n_components, cov_type):
    logger.info('Clustering GMM with n_components=%d and cov_type=%s',
                n_components, cov_type)
    gmm = GMM(n_components=n_components, covariance_type=cov_type)
    gmm.fit(X)
    labels = gmm.predict(X)
    return labels


def km_clust(X, k):
    logger.debug('Clustering KMeans with k=%d', k)
    km = KMeans(X, k)
    km.fit(X)
    labels = km.labels_
    return labels


def augment_products(products, top_n):
    # Product considered "popular" if among the top_n products ordered in its department
    top_products = products.groupby('department_id')['n_orders']\
        .nlargest(top_n).reset_index()
    top_products = top_products.join(
        products[['product_name']], on='product_id')
    top_products = top_products.values.reshape(-1)
    products['popular'] = products['product_name'].isin(
        top_products).astype(int)

    products['organic'] = products['product_name'].str.lower()\
        .str.contains('organic').astype(int)

    return products


def cluster(order_types, config):
    # strip away metadata and categoricals for clustering
    X = order_types.drop(columns=['order_dow', 'order_hour_of_day',
                                  'days_since_prior_order', 'eval_set',
                                  'order_number', 'user_id', 'size_cat']).values

    if config.get('scale_x'):
        logger.debug('Scaling order features')
        X = preprocessing.scale(X)

    # filter out rows with NaN vals
    X = clean(X)
    logger.debug('Clustering orders based on %d orders and %d variables',
                 X.shape[0], X.shape[1])

    if config.get('cluster-method') == 'gmm':
        n_components = config.get('n_components')
        cov_type = config.get('cov_type')
        labels = gmm_clust(X, n_components, cov_type)
    elif config.get('cluster-method') == 'kmeans':
        k = config.get('k')
        labels = km_clust(X, k)
    else:
        logger.warning('Invalid cluster-method set in features_config, defaulting to k-means')
        k = config.get('k')
        labels = km_clust(X, k)

    logger.debug('Cluster solution score: %f', score_ch(X, labels))
    logger.info('Cluster dispersion: %s',
                np.unique(labels, return_counts=True))

    order_types['label'] = labels
    return order_types


def build_users(orders):
    users = pd.DataFrame(orders.groupby('user_id').size().rename('n_orders'))

    # Get num orders per day of week for each user
    dow_counts = orders.pivot_table(index='user_id', columns='order_dow',
                                    values='order_number', aggfunc='count')
    dow_counts.columns = ['n_dow_{}'.format(col)
                          for col in dow_counts.columns]
    users = users.join(dow_counts.fillna(0))

    # Get num orders per hour of day for each user
    hod_counts = orders.pivot_table(index='user_id', columns='order_hour_of_day',
                                    values='order_number', aggfunc='count')
    hod_counts.columns = ['n_hod_{}'.format(col) for col in hod_counts.columns]
    users = users.join(hod_counts.fillna(0))

    # Capture each user's typical order size with summary statistics
    order_size_stats = orders.groupby('user_id').agg({
        'order_size': [np.mean, np.std, np.max, np.min]
    })
    # Flatten hierarchical column names
    order_size_stats.columns = ['_'.join(col).strip()
                                for col in order_size_stats.columns.values]

    users = users.join(order_size_stats)
    size_count = orders.pivot_table(index='user_id', columns='size_cat',
                                    values='order_number', aggfunc='count')
    users = users.join(size_count.fillna(0))

    # Describe users by the composition of their typical order
    cols = ['days_since_prior_order', 'reordered', 'organic', 'popular']
    cols += cats
    means = full.groupby('user_id')[cols].mean()
    users = users.join(means)

    return users


def get_factors(shoppers, n_components=4, random_state=903):
    """
    Find Factors to represent the shopper-level features in compressed space.
    These factors will be used to map simplified user input from application
    to the full feature space used in modeling.

    Args:
        shoppers (pd.DataFrame): full set of shoppers in feature data (train + test)
        n_components (int): number of factors to mine. Defaults to 4 and should stay that way (application
                            UI based on these 4 analyzed factors)
        random_state (int): sets random state for factor analysis algorithm. Defaults to 4 (and should stay that way)

    Returns:
        pd.DataFrame: will have n_components rows and n_features columns. The values of this matrix
                      can be used to map factors to full feature set.

    """
    # Remove columns which should not be considered in factor analysis
    for col in ['user_id', 'n_order', 'label']:
        if col in shoppers.columns:
            shoppers.drop(columns=col, inplace=True)

    # Need to scale data as columns on incommensurate scales
    x = preprocessing.scale(shoppers)
    fa = FactorAnalysis(n_components, random_state=random_state)
    fa.fit(x)
    return pd.DataFrame(fa.components_, columns=shoppers.columns)


if __name__ == '__main__':
    with open(os.path.join(ROOT, 'config', 'features_config.yml'), 'r') as f:
        config = yaml.load(f)

    if config.get('read_from') == 'local':
        data_dir = os.path.join(ROOT, 'data', 'external')
    elif config.get('read_from') == 's3':
        data_dir = 's3://{}/data/external'.format(config.get('s3_bucket-name'))
    else:
        logger.warning('Invalid read_from option in config should be one of {local, s3} '
                       'not %s. Defaulting to local', config.get('read_from'))
        data_dir = os.path.join(ROOT, 'data', 'external')
    logger.info('Reading data from %s', data_dir)

    ###########################################################################
    # Load and format data ####################################################
    ###########################################################################
    order_products = load_data(data_dir, 'order_products__train.csv')
    order_products_prior = load_data(data_dir, 'order_products__prior.csv')
    orders = load_data(data_dir, 'orders.csv')
    products = load_data(data_dir, 'products.csv')
    aisles = load_data(data_dir, 'aisles.csv')
    departments = load_data(data_dir, 'departments.csv')

    products.set_index('product_id', inplace=True)
    orders.set_index('order_id', inplace=True)
    aisles.set_index('aisle_id', inplace=True)
    departments.set_index('department_id', inplace=True)

    order_products = pd.concat([order_products, order_products_prior])
    del order_products_prior

    with open(os.path.join(ROOT, 'data', 'auxiliary', 'cats.yml'), 'r') as f:
        cat_map = yaml.load(f)
        cats = list(cat_map.keys())
    logger.debug('%d categories loaded: %s', len(cats), cats)

    ###########################################################################
    # Feature Engineering #####################################################
    ###########################################################################
    logger.debug('Engineering product features')
    n_orders = order_products.groupby('product_id').size().rename('n_orders')
    products = products.join(n_orders)
    products = augment_products(products, config.get('popular_threshold', 5))

    # cats contains mapping of macro-level category (vegetable, beverage, etc) to
    # all aisles which could fall under that classification
    for cat in cats:
        aisles[cat] = aisles['aisle'].isin(cat_map[cat]).astype(int)

    logger.debug('Engineering order features')
    orders = orders.join(
        order_products['order_id'].value_counts().rename('order_size'))

    # discretize order size to help with long-tail issue
    # these categories not currently used
    orders['size_cat'] = pd.cut(orders['order_size'],
                                [0, 5, 10, 20, np.inf],
                                labels=['small', 'medium', 'large', 'xl'])

    # Add order, product, and aisle info to every item in purchases
    full = order_products.join(orders, on='order_id')
    full = full.join(products, on='product_id')
    full = full.join(aisles, on='aisle_id')

    # order_types will be combination of metadata (from orders) and product-level features
    order_type_cols = cats + ['reordered', 'organic', 'popular']
    order_types = full.groupby('order_id')[order_type_cols].mean()
    order_types = order_types.join(orders)

    ###########################################################################
    # Clustering of Orders ####################################################
    ###########################################################################

    order_types = cluster(order_types, config)

    ###########################################################################
    # Build Users Data ########################################################
    ###########################################################################

    # Limit orders to "prior" set for training data
    logger.debug('Engineering shopper features')
    orders = orders[orders['eval_set'] == 'prior']
    shoppers = build_users(orders)

    # Add target variable (order_type for next purchase)
    # eval_set == train for each user's most recent purchase in the dataset
    user_targets = order_types[order_types['eval_set'] == 'train']
    user_targets = user_targets.groupby('user_id')['label'].first()
    shoppers = shoppers.join(user_targets.rename('label'))
    shoppers = shoppers[~shoppers.label.isna()]
    factors = get_factors(shoppers)

    shoppers_path = os.path.join(ROOT, 'data', 'features', 'shoppers.csv')
    orders_path = os.path.join(ROOT, 'data', 'features', 'baskets.csv')
    factors_path = os.path.join(ROOT, 'data', 'features', 'features.csv')

    logger.info('Feature engineering complete. Saving output...\n\t%s\n\t%s\n\t%s',
                shoppers_path, orders_path, factors_path)
    shoppers.to_csv(shoppers_path)
    order_types.to_csv(orders_path)
    factors.to_csv(factors_path)
    if config.get('upload'):
        bucket_name = config.get('s3_bucket-name')
        logger.debug('Uploading to S3 bucket: %s', bucket_name)
        s3 = boto3.client('s3')
        s3.upload_file(shoppers_path, bucket_name, 'shoppers.csv')
        s3.upload_file(orders_path, bucket_name, 'baskets.csv')
        s3.upload_file(factors_path, bucket_name, 'features.csv')
