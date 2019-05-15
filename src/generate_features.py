import logging
import os
from multiprocessing import Pool
import sys
import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score as score_ch
from sklearn.mixture import GaussianMixture as GMM


logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

ROOT = os.getcwd()
logger.debug('Project Root = %s', ROOT)

with open(os.path.join(ROOT, 'config', 'features_config.yml'), 'r') as f:
    config = yaml.load(f)
with open(os.path.join(ROOT, 'data', 'external', 'cats.yml'), 'r') as f:
    cat_map = yaml.load(f)
    cats = list(cat_map.keys())
logger.debug('%d categories loaded: %s', len(cats), cats)

if config.get('read_from') == 'local':
    data_dir = os.path.join(ROOT, 'data', 'external')
elif config.get('read_from') == 's3':
    data_dir = 's3://{}/data/external'.format(config.get('s3_bucket-name'))


def load_data(data_dir, file_name):
    path = os.path.join(data_dir, file_name)
    logger.debug('Reading data from %s', path)
    return pd.read_csv(path)


def clean(X):
    nans = np.isnan(X).any(axis=1)
    logger.debug('Removing %d rows with nan values in order features (X)',
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


def gen_plots(order_types):
    # TODO: implement heatmap plotting and save imgs
    pass


if __name__ == '__main__':
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

    n_orders = order_products.groupby('product_id').size().rename('n_orders')
    products = products.join(n_orders)

    logger.debug('Engineering product features')
    # Product considered "popular" if among the top_n products ordered in its department
    top_n = config.get('popular_threshold', 5)

    top_products = products.groupby('department_id')['n_orders']\
                   .nlargest(top_n).reset_index()
    top_products = top_products.join(products[['product_name']], on='product_id')
    top_products = top_products.values.reshape(-1)
    products['popular'] = products['product_name'].isin(top_products).astype(int)

    products['organic'] = products['product_name'].str.lower()\
                          .str.contains('organic').astype(int)

    # cats contains mapping of macro-level category (vegetable, beverage, etc) to
    # all aisles which could fall under that classification
    for cat in cats:
        aisles[cat] = aisles['aisle'].isin(cat_map[cat]).astype(int)

    logger.debug('Engineering order features')
    orders = orders.join(order_products['order_id'].value_counts().rename('order_size'))

    # discretize order size to help with long-tail issue
    # these categories not currently used
    orders['size_cat'] = pd.cut(orders['order_size'],
                                [0, 5, 10, 20, np.inf],
                                labels=['small', 'medium', 'large', 'xl'])

    full = order_products.join(orders, on='order_id')
    full = full.join(products, on='product_id')
    full = full.join(aisles, on='aisle_id')

    groups = full.groupby('order_id')

    # order_types will be combination of metadata and product-level features
    order_types = groups[cats + ['reordered', 'organic', 'popular']].mean()
    meta_cols = ['order_dow', 'order_hour_of_day', 'days_since_prior_order',
                 'eval_set', 'order_number', 'order_size', 'user_id']
    order_types = order_types.join(orders[meta_cols])
    # strip away metadata and categoricals for clustering
    X = order_types.drop(columns=['order_dow', 'order_hour_of_day',
                                  'days_since_prior_order', 'eval_set',
                                  'order_number', 'user_id']).values

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

    logger.debug('Cluster solution score: %f', score_ch(X, labels))
    logger.info('Cluster dispersion: %s',
                np.unique(labels, return_counts=True))

    order_types['label'] = labels

    if config.get('save_cluster_heatmap'):
        gen_plots(order_types)

    # Limit orders to "prior" set for training data
    logger.debug('Engineering shopper features')
    orders = orders[orders['eval_set'] == 'prior']
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
    groups = full.groupby('user_id')
    cols = ['days_since_prior_order', 'reordered', 'organic', 'popular']
    cols += cats
    means = groups[cols].mean()
    users = users.join(means)

    # Add target variable (order_type for next purchase)
    train = order_types[order_types['eval_set'] == 'train']
    user_targets = order_types[order_types['eval_set'] == 'train']
    user_targets = user_targets.groupby('user_id')['label'].first()
    users = users.join(user_targets.rename('label'))
    users = users[~users.label.isna()]

    logger.info('Feature engineering complete. Saving output...\n\t%s\n\t%s',
                shoppers_path, orders_path)
    shoppers_path = os.path.join(ROOT, 'data', 'features', 'shoppers.csv')
    orders_path = os.path.join(ROOT, 'data', 'features', 'baskets.csv')
    users.to_csv(shoppers_path)
    order_types.to_csv(orders_path)
    if config.get('upload'):
        bucket_name = config.get('s3_bucket-name')
        logger.debug('Uploading to S3 bucket: %s', bucket_name)
        s3 = boto3.client('s3')
        s3.upload_file(shoppers_path, bucket_name, 'shoppers.csv')
        s3.upload_file(orders_path, bucket_name, 'baskets.csv')
