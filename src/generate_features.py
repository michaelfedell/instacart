import argparse
import logging
import os

import boto3
import numpy as np
import pandas as pd
import yaml
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score as score_ch
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import FactorAnalysis
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


cluster_methods = {
    'kmeans': KMeans,
    'gmm': GaussianMixture
}

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# File should be run from top level dir in project via `make features`
ROOT = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
logger.debug('Project Root = %s', ROOT)


def load_data(data_dir):
    """
    Load all external data for cleaning/augmentation

    Args:
        data_dir: path to directory for raw data

    Returns:
        dict: data dictionary of pd.DataFrame with the following keys:
            - products
            - orders
            - aisles
            - departments
            - order_products

    """
    filenames = ['order_products__train', 'order_products__prior',
                 'orders', 'products', 'aisles', 'departments']
    data = {}
    for f in filenames:
        path = os.path.join(data_dir, f + '.csv')
        logger.debug('Reading data from %s', path)
        data[f] = pd.read_csv(path)

    data['products'].set_index('product_id', inplace=True)
    data['orders'].set_index('order_id', inplace=True)
    data['aisles'].set_index('aisle_id', inplace=True)
    data['departments'].set_index('department_id', inplace=True)

    data['order_products'] = pd.concat([data['order_products__train'],
                                        data['order_products__prior']])
    del data['order_products__train']
    del data['order_products__prior']

    return data


def clean(X):
    """
    Drop all rows with null values from a matrix

    Args:
        X (np.ndarray or pd.DataFrame): matrix of values to clean

    Returns:
        np.ndarray or pd.DataFrame: original matrix with rows containing null values dropped

    """
    if isinstance(X, pd.DataFrame):
        l = len(X)
        X.dropna(inplace=True)
        logger.debug('Removing %d rows with nan values in order features',
                     l - len(X))
        return X

    else:
        nans = np.isnan(X).any(axis=1)
        logger.debug('Removing %d rows with nan values in order features',
                     sum(nans))
        return X[~nans]


def augment_products(products, top_n):
    """
    Add additional product features based on frequency/name

    Args:
        products (pd.DataFrame): All products in data with n_orders column already calculated
        top_n (int): number of top orders to designate as "popular" products

    Returns:
        pd.DataFrame: original data with features appended for "popular" and "organic"

    """
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


def cluster(all_orders, config):
    """
    Perform clustering on all orders in the dataset to come up with basket archetypes

    Args:
        all_orders (pd.DataFrame): al orders in the dataset augmented with features
        config (dict): Dictionary of config read from yaml

    Returns:
        pd.DataFrame: All orders with cluster-produced labels appended

    """
    # filter out rows with NaN vals
    all_orders = clean(all_orders)
    X = all_orders
    # strip away metadata and categoricals for clustering
    for col in config.get('categoricals'):
        if col in X.columns:
            X = X.drop(columns=col)
    X = X.values

    if config.get('scale_clusters'):
        logger.debug('Scaling order features')
        X = preprocessing.scale(X)

    logger.debug('Clustering orders based on %d orders and %d variables',
                 X.shape[0], X.shape[1])

    cluster_method = config.get('cluster-method')
    logger.debug('Clustering by %s', cluster_method)
    model_class = cluster_methods.get(cluster_method)

    if not model_class:
        logger.warning('Invalid cluster-method set in features_config, defaulting to k-means')
        cluster_method = 'kmeans'
        model_class = cluster_methods.get(cluster_method)

    model = model_class(**config.get(cluster_method))
    logger.debug('Fitting the following cluster model to %d orders: \n\t%s', len(X), model)
    labels = model.fit_predict(X)
    # Start labels at 1 for readability
    labels += 1

    logger.debug('Cluster solution score: %f', score_ch(X, labels))
    logger.info('Cluster dispersion: %s',
                np.unique(labels, return_counts=True))

    all_orders.loc[:, 'label'] = labels
    return all_orders


def agg_orders(orders, col_types):
    """Aggregate orders by each cluster and define cluster characteristics"""
    amode = lambda x: int(mode(x)[0])  # return the mode value only, not counts
    # We need to replace the string "mode" in col_types dict with our lambda function for agg
    col_types = {col: (amode if t == 'mode' else t)
                 for col, t in col_types.items()}

    return orders.groupby('label').agg(col_types)


def build_shoppers(all_orders):
    """
    Build shoppers data by aggregating all the orders for each user_id

    Args:
        all_orders (pd.DataFrame): all orders in dataset with labels added

    Returns:
        pd.DataFrame: Users with features derived from order history

    """
    history = all_orders[all_orders['eval_set'] == 'prior']
    shoppers = pd.DataFrame(history.groupby('user_id').size().rename('n_orders'))
    # Get num orders per day of week for each user
    dow_counts = history.pivot_table(index='user_id', columns='order_dow',
                                     values='order_number', aggfunc='count')
    dow_counts.columns = ['n_dow_{}'.format(col)
                          for col in dow_counts.columns]
    shoppers = shoppers.join(dow_counts.fillna(0))

    # Get num orders per hour of day for each user
    hod_counts = history.pivot_table(index='user_id', columns='order_hour_of_day',
                                     values='order_number', aggfunc='count')
    hod_counts.columns = ['n_hod_{}'.format(col) for col in hod_counts.columns]
    shoppers = shoppers.join(hod_counts.fillna(0))

    # Capture each user's typical order size with summary statistics
    order_size_stats = history.groupby('user_id').agg({
        'order_size': [np.mean, np.std, np.max, np.min]
    })
    # Flatten hierarchical column names
    order_size_stats.columns = ['_'.join(col).strip()
                                for col in order_size_stats.columns]

    shoppers = shoppers.join(order_size_stats)
    size_count = history.pivot_table(index='user_id', columns='size_cat',
                                     values='order_number', aggfunc='count')
    shoppers = shoppers.join(size_count.fillna(0))

    # Describe users by the composition of their typical order
    cols = ['days_since_prior_order', 'reordered', 'organic', 'popular']
    cols += cats
    means = full.groupby('user_id')[cols].mean()
    shoppers = shoppers.join(means)

    # Add target variable (order_type for next purchase)
    # eval_set == train for each user's most recent purchase in the dataset
    targets = all_orders[all_orders['eval_set'] == 'train']
    targets = targets.groupby('user_id')['label'].first()
    logger.debug('Targets look like: \n%s', str(targets.head()))
    shoppers = shoppers.join(targets.rename('label'))

    shoppers = shoppers[~shoppers.label.isna()]

    return shoppers


def get_factors(shoppers, n_components=4, random_state=903, **kwargs):
    """
    Find Factors to represent the shopper-level features in compressed space.
    These factors will be used to map simplified user input from application
    to the full feature space used in modeling.

    Args:
        shoppers (pd.DataFrame): full set of shoppers in feature data (train + test)
        n_components (int): number of factors to mine. Defaults to 4 and should stay that way (application
                            UI based on these 4 analyzed factors)
        random_state (int): sets random state for factor analysis algorithm. Defaults to 4 (and should stay that way)
        kwargs: additional keyword arguments for sklearn.decomposition.FactorAnalysis

    Returns:
        pd.DataFrame: will have n_components rows and n_features columns. The values of this matrix
                      can be used to map factors to full feature set (on std normal scale).

    """
    # Remove columns which should not be considered in factor analysis
    x = shoppers
    for col in ['user_id', 'n_orders', 'label']:
        if col in x.columns:
            x = x.drop(columns=col)

    # Need to scale data as columns on incommensurate scales
    cols = x.columns
    x = preprocessing.scale(x)
    fa = FactorAnalysis(n_components, random_state=random_state, **kwargs)
    fa.fit(x)
    return pd.DataFrame(fa.components_, columns=cols)


def save_factor_map(factors, path):
    fig, ax = plt.subplots(figsize=(10, 15))
    sns.heatmap(factors.T, ax=ax, center=0)
    plt.savefig(path)


def plot(order_types, path):
    mpl.rcParams['figure.figsize'] = [12, 8]
    if 'label' in order_types.columns:
        order_types = order_types.set_index('label')
    sub = order_types.columns[:11]
    plot_data = pd.DataFrame(preprocessing.scale(order_types[sub]), columns=sub).T
    fig, ax = plt.subplots()
    sns.heatmap(plot_data, center=0, xticklabels=list(order_types.index), ax=ax)
    plt.xlabel('cluster label')
    plt.savefig(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate features for modeling")
    parser.add_argument('-d', '--download', default='False',
                        help='If true, will download files from S3 instead of creating from source data')
    args = parser.parse_args()
    with open(os.path.join(ROOT, 'config', 'features_config.yml'), 'r') as f:
        config = yaml.load(f)
    bucket_name = config.get('s3_bucket-name')
    if config.get('read_from') == 'local':
        data_dir = os.path.join(ROOT, 'data', 'external')
    elif config.get('read_from') == 's3':
        data_dir = 's3://{}/data/external'.format(config.get('s3_bucket-name'))
    else:
        logger.warning('Invalid read_from option in config should be one of {local, s3} '
                       'not %s. Defaulting to local', config.get('read_from'))
        data_dir = os.path.join(ROOT, 'data', 'external')

    if eval(args.download):
        data_dir = 's3://{}/'.format(config.get('s3_bucket-name'))
        logger.info('Downloading feature data from S3 bucket %s', bucket_name)

        order_types = pd.read_csv(os.path.join(data_dir, 'order_types.csv')).set_index('label')
        factors = pd.read_csv(os.path.join(data_dir, 'factors.csv'), index_col=0)
        order_types.to_csv('data/features/order_types.csv')
        factors.to_csv('data/features/factors.csv')

        logger.debug('Done with downloads')

    else:
        logger.info('Reading data from %s', data_dir)

        ###########################################################################
        # Load and format data ####################################################
        ###########################################################################
        data = load_data(data_dir)
        order_products = data['order_products']
        orders = data['orders']
        products = data['products']
        aisles = data['aisles']
        departments = data['departments']

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
        orders['size_cat'] = pd.cut(orders['order_size'], **config.get('size_cat'))

        # Add order, product, and aisle info to every item in purchases
        full = order_products.join(orders, on='order_id')
        full = full.join(products, on='product_id')
        full = full.join(aisles, on='aisle_id')

        # all_orders will be combination of product-level features and metadata (from orders)
        order_features = cats + ['reordered', 'organic', 'popular']
        all_orders = full.groupby('order_id')[order_features].mean()
        all_orders = all_orders.join(orders)

        ###########################################################################
        # Clustering of Orders ####################################################
        ###########################################################################

        all_orders = cluster(all_orders, config)
        logger.debug('Aggregating orders by label')
        order_types = agg_orders(all_orders, config.get('col_types'))

        ###########################################################################
        # Build Users Data ########################################################
        ###########################################################################

        # Limit orders to "prior" set for training data
        logger.debug('Engineering shopper features')
        shoppers = build_shoppers(all_orders)
        # n_orders is not relevant to order_type
        shoppers = shoppers.drop(columns='n_orders')
        factors = get_factors(shoppers)

        shoppers_path = os.path.join(ROOT, 'data', 'features', 'shoppers.csv')
        baskets_path = os.path.join(ROOT, 'data', 'features', 'baskets.csv')
        order_types_path = os.path.join(ROOT, 'data', 'features', 'order_types.csv')
        factors_path = os.path.join(ROOT, 'data', 'features', 'factors.csv')
        factor_map_path = os.path.join(ROOT, 'data', 'features', 'factor_map.png')

        logger.info('Feature engineering complete. Saving output...'
                    '\n\t%s\n\t%s\n\t%s\n\t%s\n\t%s',
                    shoppers_path, baskets_path, order_types_path, factors_path, factor_map_path)

        shoppers.to_csv(shoppers_path)
        all_orders.to_csv(baskets_path)
        order_types.to_csv(order_types_path)
        factors.to_csv(factors_path, index=False)
        save_factor_map(factors, factor_map_path)
        save_heatmap = config.get('save_cluster_map')
        if save_heatmap:
            logger.debug('Saving cluster heatmap to %s', save_heatmap)
            plot(order_types, save_heatmap)

        if config.get('upload'):
            logger.debug('Uploading to S3 bucket: %s', bucket_name)
            s3 = boto3.client('s3')
            s3.upload_file(shoppers_path, bucket_name, 'shoppers.csv')
            s3.upload_file(baskets_path, bucket_name, 'baskets.csv')
            s3.upload_file(order_types_path, bucket_name, 'order_types.csv')
            s3.upload_file(factors_path, bucket_name, 'factors.csv')
