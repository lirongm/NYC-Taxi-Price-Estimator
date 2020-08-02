import logging
import numpy as np
import pandas as pd
from numbers import Number
from sklearn.model_selection import train_test_split

from src.helpers import read_csv, load_yaml, write_csv

logger = logging.getLogger(__name__)

# get a subset of dataset for training and testing stratified on desired columns
def stratified_sampling(df, strata_cols=['pickup_hour', 'pickup_dayofweek'], sample_obs=50000, test_size=0.3,
                        random_state=678):
    """Perform stratified sampling on multiple columns and generate train and test sets based on strata

    Args:
        df (`pandas.DataFrame`): The data frame that contains all data that need to sample from.
        strata_cols (`list` of str): A list of variable names that the sampling is stratified on.
            Default: ['pickup_hour', 'pickup_dayofweek']
        sample_obs (int): The total number of observations in training and test sets desired. Default:50000.
        test_size (int): The proportion of test size out of the total sample. Default: 0.3.
        random_state (int): Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls. Default: 678.

    Returns:
        train_df (`pandas.DataFrame`): The data frame that contains training set.
        test_df (`pandas.DataFrame`): The data frame that contains test set.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("The `df` input has to be pd.DataFrame")

    if not all(isinstance(x, Number) for x in [sample_obs, test_size, random_state]):
        raise TypeError("At least one of inputs (sample_obs, test_size, random_state) is not numeric")

    # sample_obs has to be <= # of rows in df
    if df.shape[0] < sample_obs:
        sample_obs = df.shape[0]

    # check strata_cols type
    # create a `strata` column
    if isinstance(strata_cols, str):
        df['strata'] = df[strata_cols]
    elif isinstance(strata_cols, list):
        if len(strata_cols) == 1:
            df['strata'] = df[strata_cols[0]]
        else:
            # create a strata column and concatenate multiple columns that need to be stratified on
            df['strata'] = ''

            for col in strata_cols:
                try:
                    df['strata'] = df['strata'] + df[col].astype(str)
                except KeyError:
                    logger.error("Failed to perform stratified sampling. %s in strata_cols does not exist in data "
                                 "frame." % col)
    else:
        raise TypeError("The strata_cols in stratified_sampling() can only be str or list, but %s is obtained"
                        % type(strata_cols))

    # create data frames to save train and test sets
    train_df = pd.DataFrame(columns=df.columns)
    test_df = pd.DataFrame(columns=df.columns)

    # calculate approximate # of obs for each strata
    strata_obs = int(np.ceil(sample_obs / len(df['strata'].unique())))

    # for each strata
    for strata in df['strata'].unique():

        # get all observations in this strata
        subset = df.loc[df['strata'] == strata, :]

        # if there are no obs in this strata, nothing would be added to training/test set
        if subset.shape[0] == 0:
            pass

        # if there is only one observation in this strata, add it to training set
        elif subset.shape[0] == 1:
            train_df = train_df.append(subset, ignore_index=True, sort=False)

        # if there are more than one observation in this strata
        else:
            # use all observations if total # of obs in this strata  <= # of obs we need to get for each strata
            if subset.shape[0] < strata_obs:
                # split all obs in this strata into train and test
                train, test = train_test_split(subset, test_size=test_size, random_state=random_state)

            # if total # of rows > # of observations we want to get
            else:
                # get the pre-defined # of obs for this strata
                subset = subset.sample(n=strata_obs, random_state=random_state)
                # split all obs in this strata into train and test
                train, test = train_test_split(subset, test_size=test_size, random_state=random_state)

            # add the train and test set obtained from this strata to df that contain all train and test samples
            train_df = train_df.append(train, ignore_index=True, sort=False)
            test_df = test_df.append(test, ignore_index=True, sort=False)

    # drop 'strata' column afterwards
    train_df.drop(['strata'], axis=1, inplace=True)
    test_df.drop(['strata'], axis=1, inplace=True)

    logger.info('Train test split has been done. The training and test sets contains %i and %i observations, '
                'respectively.' % (train_df.shape[0], test_df.shape[0]))
    return train_df, test_df


def one_hot_encoder(data, one_hot_dict):
    """One hot encode a variable in a data frame

    Args:
        data (`pandas.DataFrame`): The data frame that contains the variable needs to be one-hot encoded
        one_hot_dict (:obj:`list` of :obj:`dict`):
            A list of dictionaries containing name and all possible values for features that need to be one-hot encoded.
            For each dict, key = feature name, value = a list of all possible values for the feature
    Returns:
        data (`pandas.DataFrame`): The data frame after encoding the specified variable. The column names of binary
            variables are in the format of `<original variable name>_<category>`
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("The `data` input has to be pd.DataFrame")

    if isinstance(vars, dict):
        raise TypeError("The `one_hot_dict` input has to be dict")

    # for each feature that needs to be one-hot encoded
    for feature in one_hot_dict:
        if feature not in list(data.columns):
            raise KeyError("At least one column needs to be one-hot encoded does not exist in data frame")

        # generate binary indicators corresponding to feature value and fill in 1 for all rows
        data = pd.concat([data, pd.get_dummies(data[feature], prefix=feature)], axis=1)

        # check whether data does not contain any observations for some values
        # need to generate those columns and fill in zeros to make train and test sets consistent with dimensions
        for value in one_hot_dict[feature]:
            name = feature + '_' + str(value)
            if name not in list(data.columns):
                data[name] = 0

        # drop the original feature
        data.drop([feature], axis=1, inplace=True)
    return data

def run_split(args):
    """ Wrapper function to pass in args, load configuration, read data and execute each step to generate train
    and test sets"""

    logger.info("------------------Starting to generate train and test sets------------------")
    config = load_yaml(args.config)
    config_split = config['split']
    data = read_csv(args.input)
    df_train, df_test = stratified_sampling(data, **config_split['stratified_sampling'])
    df_train = one_hot_encoder(df_train, **config_split['one_hot_encoder'])
    df_test = one_hot_encoder(df_test, **config_split['one_hot_encoder'])
    write_csv(df_train, args.output_train, description='Training set')
    write_csv(df_test, args.output_test, description='Test set')
    logger.info("------------------Finished generating train and test sets------------------")
