import os
import boto3
import logging


logger = logging.getLogger(__name__)


def s3_upload(args):
    """Upload a file to an S3 object.

	Args:
		args: Argparse args - should include args.file_local_path, args.s3_bucket_name, args.s3_path

	Returns:
		none
	"""

    # get credentials from environment variables
    logger.info("Uploading %s to S3", args.file_local_path)
    s3 = boto3.client('s3', aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))
    s3.upload_file(args.file_local_path, args.s3_bucket_name, args.s3_path)
    logger.info("Uploaded to %s in %s bucket" % (args.s3_path, args.s3_bucket_name))
