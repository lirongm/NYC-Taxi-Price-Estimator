import os
import logging

import sqlalchemy as sql
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float

from src.helpers import check_path

logger = logging.getLogger(__name__)

Base = declarative_base()

class Prediction(Base):

    """Create a data model for the database to be set up for the table `prediction` """

    __tablename__ = 'prediction'

    id = Column(Integer, primary_key=True, unique=False, nullable=False)
    pickup_longitude = Column(Float, unique=False, nullable=False)
    pickup_latitude = Column(Float, unique=False, nullable=False)
    dropoff_longitude = Column(Float, unique=False, nullable=False)
    dropoff_latitude = Column(Float, unique=False, nullable=False)
    passenger_count = Column(Integer, unique=False, nullable=False)
    pickup_hour = Column(Integer, unique=False, nullable=False)
    pickup_dayofweek = Column(String(100), unique=False, nullable=False)
    predicted_fare = Column(Float, unique=False, nullable=False)

    def __repr__(self):
        pred_repr = "<Prediction(id='%d', predicted_fare='%f')>"
        return pred_repr % (self.id, self.predicted_fare)

def create_connection(RDS=False, engine_string=None):
    """Create MySQL connection locally or in RDS"""

    if RDS:
        # Generate SQLALCHEMY_DATABASE_URI using environment variables for RDS
        MYSQL_USER = os.environ.get('MYSQL_USER')
        MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD')
        MYSQL_HOST = os.environ.get('MYSQL_HOST')
        MYSQL_PORT = os.environ.get('MYSQL_PORT')
        DATABASE_NAME = os.environ.get('DATABASE_NAME')
        MYSQL_DIALECT = 'mysql+pymysql'
        SQLALCHEMY_DATABASE_URI = os.environ.get('SQLALCHEMY_DATABASE_URI')

        if SQLALCHEMY_DATABASE_URI is not None:
            engine_string = SQLALCHEMY_DATABASE_URI
        elif (not MYSQL_DIALECT) | (not MYSQL_USER) | (not MYSQL_PASSWORD) | (not MYSQL_HOST) | (not MYSQL_PORT) |\
            (not DATABASE_NAME):
            logger.error("Failed to connect to RDS. At least one required environment variable is missing")
        else:
            engine_string = '{dialect}://{user}:{pw}@{host}:{port}/{db}'.format(dialect=MYSQL_DIALECT,
                                                                                user=MYSQL_USER,
                                                                                pw=MYSQL_PASSWORD,
                                                                                host=MYSQL_HOST,
                                                                                port=MYSQL_PORT,
                                                                                db=DATABASE_NAME)

    elif engine_string is None:
        logger.error('The path to a local database has to be specified to create connection')

    engine = sql.create_engine(engine_string)
    logger.info('Set up MySQL connection successfully')

    return engine


def create_local_db(args):
    """Creates a local database with the data model given by obj:`Prediction`

    Args:
        args: Argparse args which includes args.engine_string

    Returns: None
    """

    if args.engine_string is None:
        args.engine_string = 'sqlite:///data/prediction.db'

    if 'sqlite' not in args.engine_string:
        logger.warning("engine_string does not contain `sqlite`. Connecting to RDS.")

    else:
        # make sure the directory does exist and create one if it does not exist
        check_path(args.engine_string.strip('sqlite:///'))

    engine = create_connection(RDS=False, engine_string=args.engine_string)
    Base.metadata.create_all(engine)
    logger.info('`prediction` table has been created at %s' % args.engine_string)

def create_RDS_db(args):
    """Creates a database with the data model given by obj:`Prediction` in RDS

    Args:
        args: Argparse args which includes args.engine_string

    Returns: None
    """
    # if engine_string is not given, get it by concatenating corresponding environment variables
    if args.engine_string is None:
        engine = create_connection(RDS=True)
    else:
        engine = create_connection(RDS=True, engine_string=args.engine_string)
    Base.metadata.create_all(engine)
    logger.info('`prediction` table has been created in RDS')

