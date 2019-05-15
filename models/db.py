from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, MetaData
import sqlalchemy as sql
import logging
import os
import pandas as pd
from scipy.stats import mode
import argparse

Base = declarative_base()

# set up looging config
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DATABASE_NAME = 'instacart'
SQLITE_DB_STRING = 'sqlite:///data/{}.db'.format(DATABASE_NAME)

conn_type = 'mysql+pymysql'
user = os.environ.get('MYSQL_USER')
password = os.environ.get('MYSQL_PASSWORD')
host = os.environ.get('MYSQL_HOST')
port = os.environ.get('MYSQL_PORT')
RDS_DB_STRING = '{}://{}:{}@{}:{}/{}'\
                .format(conn_type, user, password, host, port, DATABASE_NAME)


class Shopper(Base):
    """Create a data model for shoppers in the instacart data"""
    __tablename__ = 'shoppers'

    id = Column(Integer, primary_key=True)
    order_size_mean = Column(Float, unique=False, nullable=False)
    order_size_std = Column(Float, unique=False, nullable=False)
    order_size_amax = Column(Integer, unique=False, nullable=False)
    order_size_amin = Column(Integer, unique=False, nullable=False)
    small = Column(Integer, unique=False, nullable=False)
    medium = Column(Integer, unique=False, nullable=False)
    large = Column(Integer, unique=False, nullable=False)
    xl = Column(Integer, unique=False, nullable=False)
    days_since_prior_order = Column(Float, unique=False, nullable=False)
    reordered = Column(Float, unique=False, nullable=False)
    organic = Column(Float, unique=False, nullable=False)
    popular = Column(Float, unique=False, nullable=False)
    prepared = Column(Float, unique=False, nullable=False)
    dairy = Column(Float, unique=False, nullable=False)
    gluten = Column(Float, unique=False, nullable=False)
    snack = Column(Float, unique=False, nullable=False)
    meat = Column(Float, unique=False, nullable=False)
    fish = Column(Float, unique=False, nullable=False)
    beverage = Column(Float, unique=False, nullable=False)
    veg = Column(Float, unique=False, nullable=False)
    label = Column(Integer, unique=False, nullable=False)

    def __repr__(self):
        return '<Shopper %s>' % self.id


class OrderType(Base):
    """Create a data model for order types derived from cluster centroids."""
    __tablename__ = 'ordertypes'

    label = Column(Integer, primary_key=True)

    # Described by means
    reordered = Column(Float, unique=False, nullable=False)
    organic = Column(Float, unique=False, nullable=False)
    popular = Column(Float, unique=False, nullable=False)
    prepared = Column(Float, unique=False, nullable=False)
    dairy = Column(Float, unique=False, nullable=False)
    gluten = Column(Float, unique=False, nullable=False)
    snack = Column(Float, unique=False, nullable=False)
    meat = Column(Float, unique=False, nullable=False)
    fish = Column(Float, unique=False, nullable=False)
    beverage = Column(Float, unique=False, nullable=False)
    veg = Column(Float, unique=False, nullable=False)
    order_size = Column(Float, unique=False, nullable=False)

    # Described by modes
    order_dow = Column(Integer, unique=False, nullable=False)
    order_hour_of_day = Column(Integer, unique=False, nullable=False)
    days_since_prior_order = Column(Integer, unique=False, nullable=False)
    size = Column(Integer, unique=False, nullable=False)

    def __repr__(self):
        return '<OrderType %s>' % self.label


def run_ingest(engine_string, shoppers_path, baskets_path):
    shoppers = pd.read_csv(shoppers_path)
    orders = pd.read_csv(baskets_path)

    logger.debug('Aggregating orders by label')
    order_types = orders.groupby('label').agg({
        'reordered':    'mean',
        'organic':      'mean',
        'popular':      'mean',
        'prepared':     'mean',
        'dairy':        'mean',
        'gluten':       'mean',
        'snack':        'mean',
        'meat':         'mean',
        'fish':         'mean',
        'beverage':     'mean',
        'veg':          'mean',
        'order_size':   'mean',

        'order_dow':              mode,
        'order_hour_of_day':      mode,
        'days_since_prior_order': mode,
        'order_size':             mode
    })

    logger.info('Connecting to: %s', engine_string)
    engine = sql.create_engine(engine_string)

    logger.info('Writing %d shoppers to database', len(shoppers))
    shoppers.to_sql('shoppers', engine, if_exists='append')

    logger.info('Writing %d order types to database', len(order_types))
    order_types.to_sql('ordertypes', engine, if_exists='append')

    logger.info('Done!')


def run_build(args):
    if args.mode == 'local':
        engine_string = SQLITE_DB_STRING
    elif args.mode == 'rds':
        engine_string = RDS_DB_STRING
        if (user is None or password is None or
                host is None or port is None):
            logger.error('MYSQL environment vars not specified. Be sure to '
                         '`export MYSQL_XXX=YYY` for XXX {USER, PASSWORD, HOST, PORT}')
            logger.info('Defaulting to local sqlite file')
            engine_string = SQLITE_DB_STRING
    logger.info('Connecting to: %s', engine_string)
    engine = sql.create_engine(engine_string)
    Base.metadata.create_all(engine)
    logger.info('Tables Created for %s', list(Base.metadata.tables.keys()))

    if args.populate:
        logger.debug('Running Ingestion Process')
        run_ingest(engine_string, args.shoppers, args.baskets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Create a database with the appropriate tables")
    parser.add_argument('--mode', default='local',
                        help='Can be either "local" or "rds" (will create sqlite or mysql)')
    parser.add_argument('--populate', default=False,
                        help='Will fill database with features if True')
    parser.add_argument('--shoppers', default='data/features/shoppers.csv',
                        help='Path to shoppers.csv file')
    parser.add_argument('--baskets', default='data/features/baskets.csv',
                        help='Path to baskets.csv file')

    args = parser.parse_args()

    run_build(args)