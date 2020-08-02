import logging
import numpy as np
import pandas as pd
from numbers import Number

from src.helpers import read_csv, load_yaml, write_csv

logger = logging.getLogger(__name__)

def remove_missing_obs(df):
    """Remove observations with any missing values and return the cleaned data frame"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The `df` input has to be pd.DataFrame")

    df = df.dropna(axis=0)
    logger.info("Observations with any missing values have been dropped")
    return df


def clean_key(df):
    """Remove `key` column to reduce the data size. This column is unnecessary"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The `df` input has to be pd.DataFrame")

    if 'key' not in list(df.columns):
        logger.warning("Failed to drop feature `key`: it does not exist in df. Original df has been returned.")
    else:
        df.drop(['key'], axis=1, inplace=True)
        logger.info("`key` column has been dropped.")
    return df


def clean_fare_amount(df, initial_charge=2.5):
    """Clean fare_amount by dropping observations whose fare_amount < initial charge of NYC taxi.

    Args:
        df (`pandas.DataFrame`): The data frame that needs to be cleaned
        initial_charge (int_or_float): The initial charge of NYC Taxi Fare. Default: 2.5 (according
            to: https://www1.nyc.gov/site/tlc/passengers/taxi-fare.page)

    Returns:
        df (`pandas.DataFrame`): The cleaned data frame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The `df` input has to be pd.DataFrame")

    if not isinstance(initial_charge, Number):
        raise TypeError("The `initial_charge` input has to be numeric")

    if initial_charge <= 0:
        raise ValueError("The initial charge for fare amount has to be positive")

    if 'fare_amount' not in list(df.columns):
        logger.warning("Failed to drop feature `fare_amount`: it does not exist in df. Original df has been returned.")
    else:
        df.drop(index=df[(df.fare_amount < initial_charge)].index, inplace=True)
        logger.info("`fare_amount` has been cleaned.")
    return df


def clean_locations(df, nyc_min_lon=-75, nyc_max_lon=-72, nyc_min_lat=39, nyc_max_lat=42):
    """Clean longitude and latitude features by dropping observations with drop-off or pickup locations beyond NYC

    Args:
        df (`pandas.DataFrame`): The data frame that needs to be cleaned
        nyc_min_lon: The minimum longitude of NYC
        nyc_max_lon: The maximum longitude of NYC
        nyc_min_lat: The minimum latitude of NYC
        nyc_min_lat: The minimum latitude of NYC

    Returns:
        df (`pandas.DataFrame`): The cleaned data frame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The `df` input has to be pd.DataFrame")

    if not all(isinstance(x, Number) for x in [nyc_min_lon, nyc_max_lon, nyc_min_lat, nyc_max_lat]):
        raise TypeError("At least one longitude and latitude input is not numeric")

    # check field exists and are numeric
    if ('pickup_longitude' or 'dropoff_longitude' or 'pickup_latitude' or 'dropoff_latitude') not in list(df.columns):
        raise KeyError("Data does not contain a required location field. Columns in data are %s" % df.columns.to_list())

    for col in ['pickup_longitude', 'dropoff_longitude', 'pickup_latitude', 'dropoff_latitude']:
        if not np.issubdtype(df[col].dtype, np.number):
            raise TypeError("%s has to be numeric" % col)

    df.drop(index=df[(df.pickup_longitude < nyc_min_lon)
                     | (df.pickup_longitude > nyc_max_lon)
                     | (df.dropoff_longitude < nyc_min_lon)
                     | (df.dropoff_longitude > nyc_max_lon)
                     | (df.pickup_latitude < nyc_min_lat)
                     | (df.pickup_latitude > nyc_max_lat)
                     | (df.dropoff_latitude < nyc_min_lat)
                     | (df.dropoff_latitude > nyc_max_lat)].index, inplace=True)
    logger.info("Location features have been cleaned.")

    return df


def clean_passenger_count(df, min_count=1, max_count=5):
    """Clean passenger_count by dropping observations with passenger_count beyond the range defined

    Args:
        df (`pandas.DataFrame`): The data frame that needs to be cleaned
        min_count (int): The minimum number of passengers in a record for it to be considered as valid. Optional. Default = 1.
        max_count (int): The maximum number of passengers in a record for it to be considered as valid. Optional. Default = 5.

    Returns:
        df (`pandas.DataFrame`): The cleaned data frame
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The `df` input has to be pd.DataFrame")

    if not all(isinstance(x, int) for x in [min_count, max_count]):
        raise TypeError("At least one count inputs is not an integer")

    if (min_count <= 0) | (min_count > max_count):
        raise ValueError("min_count has to be positive and no greater than max_count")

    df.drop(index=df[(df.passenger_count < min_count) | (df.passenger_count > max_count)].index, inplace=True)
    logger.info("`passenger_count` has been cleaned.")
    return df


def run_clean(args):
    """ Wrapper function to pass in args, load configuration, read data, and execute each step in data cleaning """

    logger.info("------------------Starting to clean data-----------------")
    # read data and configuration
    data = read_csv(args.input)
    config = load_yaml(args.config)
    config_clean = config['clean']

    # remove observations with any missing values
    nonmissing_data = remove_missing_obs(data)

    # clean each feature
    data = clean_key(nonmissing_data)
    data = clean_fare_amount(data, **config_clean['clean_fare_amount'])
    data = clean_locations(data, **config_clean['clean_locations'])
    data = clean_passenger_count(data, **config_clean['clean_passenger_count'])

    # save output
    write_csv(data, path=args.output, description='Clean data')

    logger.info("------------------Finished cleaning data-----------------")
