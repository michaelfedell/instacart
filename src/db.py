import argparse
import logging
import os
import sys

import pandas as pd
import sqlalchemy as sql
from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
sys.path.append(os.path.dirname(sys.path[0]))  # so that config can be imported from project root
import config
import yaml

Base = declarative_base()

# set up looging config
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

with open(os.path.join('config', 'features_config.yml'), 'r') as f:
    col_types = yaml.load(f).get('col_types')


class OrderType(Base):
    """Create a data model for order types derived from cluster centroids.

    Each of these rows will describe one of the order types derived from
    clustering during the feature generation process. An order type is
    described by its centroid for the most part. Temporal features and order_size
    are defined by the mode of the cluster since most common hour of day is more
    interesting than the average of all times (same logic for other mode values).
    """
    __tablename__ = 'ordertypes'
    # We use the column_types (mean or mode) to determine if column should be stored as int or float
    col_types = {col: (Integer if t == 'mode' else Float)
                 for col, t in col_types.items()}

    index = Column(Integer, primary_key=True)
    label = Column(Integer, unique=False, nullable=False)
    # Described by means
    reordered = Column(col_types.get('reordered', Float), unique=False, nullable=False)
    organic = Column(col_types.get('organic', Float), unique=False, nullable=False)
    popular = Column(col_types.get('popular', Float), unique=False, nullable=False)
    prepared = Column(col_types.get('prepared', Float), unique=False, nullable=False)
    dairy = Column(col_types.get('dairy', Float), unique=False, nullable=False)
    gluten = Column(col_types.get('gluten', Float), unique=False, nullable=False)
    snack = Column(col_types.get('snack', Float), unique=False, nullable=False)
    meat = Column(col_types.get('meat', Float), unique=False, nullable=False)
    fish = Column(col_types.get('fish', Float), unique=False, nullable=False)
    beverage = Column(col_types.get('beverage', Float), unique=False, nullable=False)
    veg = Column(col_types.get('veg', Float), unique=False, nullable=False)

    # Described by modes
    order_dow = Column(col_types.get('order_dow', Float), unique=False, nullable=False)
    order_hour_of_day = Column(col_types.get('order_hour_of_day', Float), unique=False, nullable=False)
    days_since_prior_order = Column(col_types.get('days_since_prior_order', Float), unique=False, nullable=False)
    order_size = Column(col_types.get('order_size', Float), unique=False, nullable=False)

    # Descriptions will be populated by hand upon cluster examination
    desc = Column(String(240), nullable=True)

    def __repr__(self):
        return '<OrderType %s>' % self.label


def run_ingest(engine_string, order_types_path):
    """
    Create db if needed and populate with data

    Args:
        engine_string (str): Connection string to use
        order_types_path (str): Path to order_types csv describing centroids

    Returns:

    """
    order_types = pd.read_csv(order_types_path)

    logger.info('Connecting to: %s', engine_string)
    engine = sql.create_engine(engine_string)

    logger.info('Writing %d order types to database', len(order_types))
    order_types.index = order_types.index.astype(int)
    order_types.to_sql('ordertypes', engine, if_exists='append')

    logger.info('Done!')


def run_build(args):
    """Create the database with ordertypes table"""
    if args.mode == 'local':
        engine_string = config.SQLITE_DB_STRING
    elif args.mode == 'rds':
        engine_string = config.RDS_DB_STRING
        # Default to local if any required env vars are missing
        if (config.user is None or config.password is None or
                config.host is None or config.port is None):
            logger.error('MYSQL environment vars not specified. Be sure to '
                         '`export MYSQL_XXX=YYY` for XXX {USER, PASSWORD, HOST, PORT}')
            logger.info('Defaulting to local sqlite file')
            engine_string = config.SQLITE_DB_STRING
    else:
        logger.warning('%s is not a valid mode, defaulting to local', args.mode)
        engine_string = config.SQLITE_DB_STRING

    logger.info('Connecting to: %s', engine_string)
    engine = sql.create_engine(engine_string)
    Base.metadata.create_all(engine)
    logger.info('Tables Created for %s', list(Base.metadata.tables.keys()))

    if args.populate:
        logger.debug('Running Ingestion Process')
        run_ingest(engine_string, args.ordertypes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a database with the appropriate tables")
    parser.add_argument('--mode', default='local', choices=['local', 'rds'],
                        help='Can be either "local" or "rds" (will create sqlite or mysql)')
    parser.add_argument('--populate', action='store_true',
                        help='Will fill database with features if included')
    parser.add_argument('--ordertypes', default='data/features/order_types.csv',
                        help='Path to order_types.csv file')

    args = parser.parse_args()

    run_build(args)
