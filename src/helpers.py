import os
import yaml
import pickle
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def read_csv(path):
    """Read csv from a given path"""

    try:
        df = pd.read_csv(path)
        logger.info('Input data loaded from %s', path)
        return df
    except FileNotFoundError:
        logger.error("%s is invalid. Please provide a valid file location to read csv from." % path)
    except Exception as e:
        logger.error(e)


def write_csv(output, path, description=None, index=False):
    """Write a pd.DataFrame as csv to a given path"""

    # check the path is valid
    check_path(path)

    try:
        output.to_csv(path, index=index)

        # if a file description is given, use that in logging
        if description is None:
            logger.info("Output saved to %s" % path)
        else:
            logger.info("%s saved to %s" % (description, path))
    except Exception as e:
        logger.error(e)


def check_path(path):
    """Create a directory if a directory or the directory of a file does not exist"""

    # check whether path contains a file name
    # extract the directory if path contains a file name
    if os.path.basename(path) != '':
        path = os.path.dirname(path)

    # if the directory does not exist, create it
    if os.path.exists(path) is False:
        os.makedirs(path)
        logger.info("The path %s does not exist and has been created." % path)

def load_yaml(path):
    """ Load a yaml file from a given path"""
    try:
        # Load configuration file
        with open(path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        logger.info("Configuration file loaded from %s" % path)
        return config
    except FileNotFoundError:
        logger.error("%s is invalid. Please provide a valid file location to load yaml file from." % path)
    except Exception as e:
        logger.error(e)

def load_model(path):
    """ Load a trained model object from a given path"""
    try:
        model = pickle.load(open(path, 'rb'))
        logger.info("Model has been loaded from %s" % path)
        return model
    except FileNotFoundError:
        logger.error("%s is invalid. Please provide a valid path to read the model from." % path)
    except Exception as e:
        logger.error(e)

