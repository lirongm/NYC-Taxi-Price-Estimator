import boto3
import logging

from src.helpers import check_path

logger = logging.getLogger(__name__)

def download(save_to, s3_bucket, s3_key):
    """Download data fram S3

    Args:
        save_to (`str`): The path to save the file locally
        s3_bucket (`str`): The name of the bucket to download from
        s3_key (`str`): The name of the key to download from
    Returns:
        None
    """
    try:
        # check if path is valid and properly handled it if invalid
        check_path(save_to)

        logger.info('Downloading raw data from S3')
        s3 = boto3.resource('s3')
        s3.meta.client.download_file(s3_bucket, s3_key, save_to)
    except Exception as e:
        logger.error("Failed to download raw data from S3, since %s" % e)

def run_download(args):
    """Load configuration file and pass argparse args which include args.output """
    logger.info("Please note it takes a while to download the raw data (~5GB) from S3")
    download(args.output, args.s3_bucket, args.s3_key)
