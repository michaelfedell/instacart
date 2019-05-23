import logging
import os
import argparse
import glob
import boto3

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_file_names(top_dir, file_expr):
    """Get all file names in a directory subtree
    Args:
        top_dir (str): The base directory from which to get list_of_files from
    Returns: List of file locations
    """

    logger.debug('Searching for %s in %s', file_expr, top_dir)
    list_of_files = glob.glob(os.path.join(top_dir, file_expr), recursive=True)

    return list_of_files


def run_upload(args):
    """Loads config and executes load data set
    Args:
        args: From argparse,
            args.dir (str): Path to top-level directory
            args.file (str): Regex to find files matching via glob
            args.bucket (str): Name of S3 bucket for file upload
    Returns: None
    """
    files = get_file_names(args.dir, args.file)
    logger.info('Fetched %d files for upload', len(files))
    s3 = boto3.client('s3')
    for path in files:
        logger.debug('Uploading %s', path)
        s3.upload_file(path, args.bucket, path)
    logger.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Upload local files to S3")
    # parser.add_argument('--config', default='config/features_config.yml',
    #                     help='path to yaml file with configurations')
    parser.add_argument('--dir', default='data/external/',
                        help='Local directory to find files')
    parser.add_argument('--file', default='*.csv',
                        help='File name or file path regex')
    parser.add_argument('--bucket', default='instacart-store',
                        help='Name of S3 bucket to upload files')

    args = parser.parse_args()

    run_upload(args)
