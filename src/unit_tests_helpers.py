import pandas as pd

def format_df(df, numeric_cols='all', str_col='key', date_col='pickup_datetime'):
    """ helper function to reset index and format data frame for comparisons

    Args:
        df (`pandas.DataFrame`): data frame that needs to be formatted
        numeric_cols (`list` of `str`): list of numeric column names. Default = ['fare_amount', 'pickup_longitude',
            'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
        str_col (`str`): name of a string column
        date_col (`str`): name of a datetime column

    Returns:
        df (`pandas.DataFrame`): formatted data frame
    """

    df.reset_index(drop=True, inplace=True)

    if numeric_cols == 'all':
        numeric_cols = ['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                        'passenger_count']

    if numeric_cols is not None:
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, downcast='float')

    if str_col is not None:
        df[str_col] = df[str_col].astype(str)

    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)

    return df

def compare_df(df1, df2, numeric_cols='all', str_col='key', date_col='pickup_datetime'):
    """format and then compare two data frames"""
    return format_df(df1, numeric_cols, str_col, date_col).equals(format_df(df2, numeric_cols, str_col, date_col))

def make_raw_data():
    """make data in the format of raw_data.csv to test functions"""
    df = pd.DataFrame([
        ['2010-01-05 16:52:16.0000002', 16.9, '2010-01-05 16:52:16 UTC', -74.016048, 40.711303, -73.979268, 40.782004,
         1],
        ['2009-06-15 17:26:21.0000001', 4.5, '2009-06-15 17:26:21 UTC', -73.844311, 40.721319, -73.84161, 40.712278,
         1]
        ],
        columns=['key', 'fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
                 'dropoff_longitude', 'dropoff_latitude', 'passenger_count'])
    return df

def make_clean_data():
    """make data in the format of clean-data.csv to test functions"""
    df = pd.DataFrame([
        [16.9, '2010-01-05 16:52:16', -74.016048, 40.711303, -73.979268, 40.782004, 1],
        [4.5, '2009-06-15 17:26:21', -73.844311, 40.721319, -73.84161, 40.712278, 1]
    ],
        columns=['fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude',
                 'dropoff_longitude', 'dropoff_latitude', 'passenger_count'])
    return df

def make_features_data():
    """make data in the format of features-data.csv to test functions"""
    df = pd.DataFrame([
        [22.54, -74.01048278808595, 40.71766662597656, -73.98577117919923, 40.66036605834961, 1, 5, 'Sunday', 0.06240],
        [58.0, -73.98332977294923, 40.73871994018555, -73.93319702148438, 40.84722518920898, 1, 5, 'Sunday', 0.119526],
        [4.5, -73.99017333984375, 40.756446838378906, -73.9856185913086, 40.7628288269043, 1, 10, 'Thursday', 0.007840],
        [5.0, -73.95479583740234, 40.779335021972656, -73.94493103027344, 40.780086517333984, 1, 10, 'Thursday', 0.0098]
    ],
        columns = ['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                   'passenger_count', 'pickup_hour', 'pickup_dayofweek', 'distance'])
    return df

def make_train_data():
    """make data in the format of train-data.csv to test functions"""
    df = pd.DataFrame([
        [4.5, -73.94586944580078, 40.77777862548828, -73.95226287841797, 40.77225112915039, 1, 0.008451579520770735, 0,
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [8.5, -74.0005111694336, 40.737468719482415, -73.99774932861328, 40.75411605834961, 3, 0.01687488240184069, 0, 
         0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    ],
        columns = ['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                   'passenger_count', 'distance', 'pickup_dayofweek_Friday', 'pickup_dayofweek_Monday',
                   'pickup_dayofweek_Saturday', 'pickup_dayofweek_Sunday', 'pickup_dayofweek_Thursday',
                   'pickup_dayofweek_Tuesday', 'pickup_dayofweek_Wednesday', 'pickup_hour_0', 'pickup_hour_1',
                   'pickup_hour_2', 'pickup_hour_3', 'pickup_hour_4', 'pickup_hour_5', 'pickup_hour_6',
                   'pickup_hour_7', 'pickup_hour_8', 'pickup_hour_9', 'pickup_hour_10', 'pickup_hour_11',
                   'pickup_hour_12', 'pickup_hour_13', 'pickup_hour_14', 'pickup_hour_15', 'pickup_hour_16',
                   'pickup_hour_17', 'pickup_hour_18', 'pickup_hour_19', 'pickup_hour_20', 'pickup_hour_21',
                   'pickup_hour_22', 'pickup_hour_23']
    )
    return df

def make_test_data():
    """make data in the format of test-data.csv to test functions"""
    df = pd.DataFrame([
        [52.0, -73.78187561035155, 40.64474487304688, -73.95035552978516, 40.76160049438477, 1, 0.20503833663639467, 
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [5.0, -73.98230743408203, 40.767921447753906, -73.98240661621094, 40.77505111694336, 1, 0.007130359026425778,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    ],
        columns = ['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                   'passenger_count', 'distance', 'pickup_dayofweek_Friday', 'pickup_dayofweek_Monday',
                   'pickup_dayofweek_Saturday', 'pickup_dayofweek_Sunday', 'pickup_dayofweek_Thursday',
                   'pickup_dayofweek_Tuesday', 'pickup_dayofweek_Wednesday', 'pickup_hour_0', 'pickup_hour_1',
                   'pickup_hour_2', 'pickup_hour_3', 'pickup_hour_4', 'pickup_hour_5', 'pickup_hour_6',
                   'pickup_hour_7', 'pickup_hour_8', 'pickup_hour_9', 'pickup_hour_10', 'pickup_hour_11',
                   'pickup_hour_12', 'pickup_hour_13', 'pickup_hour_14', 'pickup_hour_15', 'pickup_hour_16',
                   'pickup_hour_17', 'pickup_hour_18', 'pickup_hour_19', 'pickup_hour_20', 'pickup_hour_21',
                   'pickup_hour_22', 'pickup_hour_23']
    )
    return df

def make_pred_data():
    """make data in the format of test-predictions.csv to test functions"""
    df = pd.DataFrame([
        [52.0, 55.40065873015904],
        [5.0, 4.7929930952380975]
    ],
        columns=['y_test', 'ypred_test']
    )
    return df
