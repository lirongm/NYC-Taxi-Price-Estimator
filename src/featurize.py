import logging
import numpy as np
import pandas as pd

from src.helpers import read_csv, load_yaml, write_csv

logger = logging.getLogger(__name__)


def generate_hour(df, generate=True):
    """Generate hour of day from pickup_datetime and return the new df, if generate is True (Default: True)"""
    if generate:

        if not isinstance(df, pd.DataFrame):
            raise TypeError("The `df` input has to be pd.DataFrame")

        if 'pickup_datetime' not in list(df.columns):
            raise KeyError("Data does not contain a pickup_datetime field. Columns in data are %s"
                           % df.columns.to_list())

        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        logger.info("`pickup_hour` column has been generated.")
    return df


def generate_dayofweek(df, generate=True):
    """Generate day of week from pickup_datetime and one-hot encode it and return the new df,
    if generate is True (Default: True)"""
    if generate:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The `df` input has to be pd.DataFrame")

        if 'pickup_datetime' not in list(df.columns):
            raise KeyError("Data does not contain a pickup_datetime field. Columns in data are %s"
                           % df.columns.to_list())

        df['pickup_dayofweek'] = df['pickup_datetime'].dt.day_name()
        logger.info("`pickup_dayofweek` column has been generated.")
    return df


def generate_distance(df, generate=True):
    """Generate distance between dropoff and pickup locations and return the new df, if generate is True (Default: True)"""
    if generate:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The `df` input has to be pd.DataFrame")

        for col in ['pickup_longitude', 'dropoff_longitude', 'pickup_latitude', 'dropoff_latitude']:
            if col not in list(df.columns):
                raise KeyError("%s does not exist in data frame" % col)

        for col in ['pickup_longitude', 'dropoff_longitude', 'pickup_latitude', 'dropoff_latitude']:
            if not np.issubdtype(df[col].dtype, np.number):
                raise TypeError("%s has to be numeric" % col)

        # calculate difference in latitude and longitude
        df["lat_diff"] = np.abs(df.dropoff_latitude - df.pickup_latitude)
        df["lon_diff"] = np.abs(df.dropoff_longitude - df.pickup_longitude)

        # calculate distance
        df['distance'] = ((df.lat_diff) ** 2 + (df.lon_diff) ** 2) ** .5

        # drop temporary variables
        df.drop(['lat_diff', 'lon_diff'], axis=1, inplace=True)
        logger.info("`distance` column has been generated.")
    return df


def run_featurize(args):
    """ Wrapper function to pass in args, load configuration, read data and execute each step in feature engineering """

    logger.info("------------------Starting to perform feature engineering-----------------")
    # read data and configuration
    data = read_csv(args.input)
    config = load_yaml(args.config)
    config_featurize = config['featurize']

    try:
        # format pickup_datetime column
        data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'], infer_datetime_format=True)

        # generate hour and day of week
        data = generate_hour(data, **config_featurize['generate_hour'])
        data = generate_dayofweek(data, **config_featurize['generate_dayofweek'])

        # drop pickup_datetime afterwards
        data.drop(['pickup_datetime'], axis=1, inplace=True)
        logger.info("`pickup_datetime` column has been dropped.")

    except KeyError:
        logger.error("Failed to generate pickup_datetime related feature: `pickup_datetime` does not exist in df. "
                     "Original df has been returned.")

    # generate distance feature
    data = generate_distance(data, **config_featurize['generate_distance'])

    # save output
    write_csv(data, path=args.output, description='Data with additional features generated')

    logger.info("------------------Finished feature engineering-----------------")
