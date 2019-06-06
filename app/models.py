import logging
from app import db

logger = logging.getLogger(__name__)

db.Model.metadata.reflect(db.engine)


class OrderType(db.Model):
    """Create a data model for the database to be set up for capturing songs
    """
    try:
        __table__ = db.Model.metadata.tables['ordertypes']
    except:
        logger.error("'ordertypes' table not found")
