import argparse
import logging.config
import os

from src.download import run_download
from src.clean import run_clean
from src.filter import run_filter
from src.featurize import run_featurize
from src.split import run_split
from src.train import run_train
from src.score import run_score
from src.evaluate import run_evaluate

from src.s3_upload import s3_upload
from src.create_db import create_local_db, create_RDS_db

logging.config.fileConfig('config/logging/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # Add parsers for both upload data to S3 and creating database
    parser = argparse.ArgumentParser(description="Upload raw data to S3 and/or create database")
    subparsers = parser.add_subparsers()

    # Sub-parser for downloading data from S3
    sb_download = subparsers.add_parser("download", description="Download raw data from S3")
    sb_download.add_argument('--s3_bucket', default='nw-lma-s3',
                             help="The S3 bucket name which contains data to download (optional, default = nw-lma-s3)")
    sb_download.add_argument('--s3_key', default='data/raw_data.csv',
                             help="The name of the key (the file path) to download from in S3 "
                                  "(optional, default = data/raw_data.csv)")
    sb_download.add_argument("--output", default='data/raw_data.csv',
                              help="The path to save the raw data downloaded from S3 "
                                   "(optional, default = data/raw_data.csv)")
    sb_download.set_defaults(func=run_download)

    # Sub-parser for filtering data
    sb_filter = subparsers.add_parser('filter', description='Filter raw data by a specific year')
    sb_filter.add_argument('--input', '-i', default='data/raw_data.csv',
                           help='Path to raw data to be filtered (optional, default = data/raw_data.csv)')
    sb_filter.add_argument('--output', default='data/filtered-data.csv',
                           help='Path to save filtered data CSV (optional, default = data/filtered-data.csv)')
    sb_filter.add_argument('--config', default='config/config.yaml',
                           help='Path to configuration file (optional, default = config/config.yaml)')
    sb_filter.set_defaults(func=run_filter)

    # Sub-parser for cleaning data
    sb_clean = subparsers.add_parser('clean', description='Clean raw data')
    sb_clean.add_argument('--input', '-i', default='data/raw_data.csv',
                           help='Path to raw data to be cleaned (optional, default = data/raw_data.csv)')
    sb_clean.add_argument('--output', default='data/clean-data.csv',
                           help='Path to save clean data CSV (optional, default = data/clean-data.csv)')
    sb_clean.add_argument('--config', default='config/config.yaml',
                           help='Path to configuration file (optional, default = config/config.yaml)')
    sb_clean.set_defaults(func=run_clean)

    # Sub-parser for generating features
    sb_featurize = subparsers.add_parser('featurize', description='Generate features')
    sb_featurize.add_argument('--input', '-i', default='data/clean-data.csv',
                           help='Path to input data (optional, default = data/clean-data.csv)')
    sb_featurize.add_argument('--output', default='data/features-data.csv',
                           help='Path to save output CSV (optional, default = data/features-data.csv)')
    sb_featurize.add_argument('--config', default='config/config.yaml',
                           help='Path to configuration file (optional, default = config/config.yaml)')
    sb_featurize.set_defaults(func=run_featurize)

    # Sub-parser for train test split
    sb_split = subparsers.add_parser('split', description='Train test split')
    sb_split.add_argument('--input', '-i', default='data/features-data.csv',
                           help='Path to input data (optional, default = data/features-data.csv)')
    sb_split.add_argument('--output_train', default='data/train-data.csv',
                           help='Path to save training set CSV (optional, default = data/train-data.csv)')
    sb_split.add_argument('--output_test', default='data/test-data.csv',
                           help='Path to save test set CSV (optional, default = data/test-data.csv)')
    sb_split.add_argument('--config', default='config/config.yaml',
                           help='Path to configuration file (optional, default = config/config.yaml)')
    sb_split.set_defaults(func=run_split)

    # Sub-parser for model training
    sb_train = subparsers.add_parser('train', description='Train a logistic regression model')
    sb_train.add_argument('--input', '-i', default='data/train-data.csv',
                           help='Path to traininig data set (optional, default = data/train-data.csv)')
    sb_train.add_argument('--output_model', default='model/model.pkl',
                           help='Path to save trained model(optional, default = model/model.pkl)')
    sb_train.add_argument('--output_feature_imp', default='evaluation/feature-imp.csv',
                           help='Path to save feature importance(optional, default = evaluation/feature-imp.csv)')
    sb_train.add_argument('--config', default='config/config.yaml',
                           help='Path to configuration file (optional, default = config/config.yaml)')
    sb_train.set_defaults(func=run_train)

    # Sub-parser for scoring model
    sb_score = subparsers.add_parser('score', description='Generate predictions on test set')
    sb_score.add_argument('--input_data', default='data/test-data.csv',
                           help='Path to test data set (optional, default = data/test-data.csv)')
    sb_score.add_argument('--input_model', default='model/model.pkl',
                           help='Path to trained model (optional, default = model/model.pkl)')
    sb_score.add_argument('--output', default='data/test-predictions.csv',
                           help='Path to save predictions CSV (optional, default = data/test-predictions.csv)')
    sb_score.add_argument('--config', default='config/config.yaml',
                           help='Path to configuration file (optional, default = config/config.yaml)')
    sb_score.set_defaults(func=run_score)

    # Sub-parser for evaluating model
    sb_evaluate = subparsers.add_parser('evaluate', description='Evaluate model performance')
    sb_evaluate.add_argument('--input', default='data/test-predictions.csv',
                           help='Path to test predictions data (optional, default = data/test-predictions.csv)')
    sb_evaluate.add_argument('--output', default='evaluation/test-metrics.txt',
                           help='Path to save evaluation metrics txt (optional, default = evaluation/test-metrics.txt)')
    sb_evaluate.add_argument('--config', default='config/config.yaml',
                           help='Path to configuration file (optional, default = config/config.yaml)')
    sb_evaluate.set_defaults(func=run_evaluate)

    # The following functionality is not going to be used in the model pipeline
    # Sub-parser for uploading data to S3
    sb_upload = subparsers.add_parser("s3_upload", description="Upload file to S3")
    sb_upload.add_argument("--file_local_path", default='data/raw_data.csv',
                           help="Local path to file to be uploaded to S3 (optional, default = data/raw_data.csv)")
    sb_upload.add_argument("--s3_bucket_name", default='nw-lma-s3',
                           help="The name of the S3 bucket to upload to (optional, default = nw-lma-s3)")
    sb_upload.add_argument("--s3_path", default='data/raw_data.csv',
                           help="The path to the file to upload to in S3 (optional, default = data/raw_data.csv)")
    sb_upload.set_defaults(func=s3_upload)

    # Sub-parser for creating a local database
    sb_create_local_db = subparsers.add_parser("create_local_db", description="Create a local database")
    sb_create_local_db.add_argument("--engine_string", default=os.environ.get('SQLALCHEMY_DATABASE_URI'),
                                    help="SQLAlchemy connection URI for a local database "
                                    "(optional, default obtained from environment variable `SQLALCHEMY_DATABASE_URI`)")
    sb_create_local_db.set_defaults(func=create_local_db)
    
    # Sub-parser for creating a RDS database
    sb_create_RDS_db = subparsers.add_parser("create_RDS_db", description="Create a database in RDS")
    sb_create_RDS_db.add_argument("--engine_string", help="SQLAlchemy connection URI for a RDS database (optional). If"
                                  "not given, URI will be obtained from environment variables")
    sb_create_RDS_db.set_defaults(func=create_RDS_db)

    args = parser.parse_args()
    args.func(args)
