import os
import pandas as pd
import numpy as np
import sklearn.ensemble
from numbers import Number
from src.unit_tests_helpers import compare_df, format_df, make_raw_data, make_clean_data, make_features_data, \
    make_train_data, make_test_data, make_pred_data
from src.filter import filter_year, process_by_chunk
from src.clean import remove_missing_obs, clean_key, clean_fare_amount, clean_locations, clean_passenger_count
from src.featurize import generate_hour, generate_dayofweek, generate_distance
from src.split import stratified_sampling, one_hot_encoder
from src.train import train_rf_model
from src.score import score_model
from src.evaluate import evaluate_model

###############
# Script: src.filter
###############

# filter df by year
def test_filter_year_happy():
    df = make_raw_data()
    filtered_df = filter_year(df, 2010)

    df_true = pd.DataFrame([
        ['2010-01-05 16:52:16.0000002',16.9,'2010-01-05 16:52:16 UTC',-74.016048,40.711303,-73.979268,40.782004,1]],
        columns=['key','fare_amount','pickup_datetime','pickup_longitude','pickup_latitude',
                 'dropoff_longitude','dropoff_latitude','passenger_count']
    )
    assert compare_df(filtered_df, df_true)

# required field does not exist
def test_filter_year_unhappy():
    df = make_raw_data()
    df.drop(['pickup_datetime'], axis=1, inplace=True)

    try:
        filter_year(df, 2010)
        assert False
    except KeyError:
        assert True

# read and filter data by chunks
def test_process_by_chunk_happy():

    if not os.path.exists('unit_tests/'):
        os.makedirs('unit_tests/')

    file_path = 'unit_tests/test_process_chunk_happy.csv'
    df = make_raw_data()
    df.to_csv(file_path, index=False)
    # read all data
    df = process_by_chunk(file_path, year=2010, max_num_rows_read=None, chunksize=1, log_per_chunks=1)

    df_true = pd.DataFrame([
        ['2010-01-05 16:52:16.0000002',16.9,'2010-01-05 16:52:16 UTC',-74.016048,40.711303,-73.979268,40.782004,1]],
        columns=['key','fare_amount','pickup_datetime','pickup_longitude','pickup_latitude',
                 'dropoff_longitude','dropoff_latitude','passenger_count']
    )

    assert compare_df(df, df_true)

# test the program will properly chunksize > max_num_rows_read
# it will reset chunksize = max_num_rows_read
def test_process_by_chunk_unhappy():
    if not os.path.exists('unit_tests/'):
        os.makedirs('unit_tests/')

    file_path = 'unit_tests/test_process_chunk_unhappy.csv'
    df = make_raw_data()
    df.to_csv(file_path, index=False)
    df = process_by_chunk(file_path, year=2010, max_num_rows_read=1, chunksize=2, log_per_chunks=1)

    df_true = pd.DataFrame([
        ['2010-01-05 16:52:16.0000002',16.9,'2010-01-05 16:52:16 UTC',-74.016048,40.711303,-73.979268,40.782004,1]],
        columns=['key','fare_amount','pickup_datetime','pickup_longitude','pickup_latitude',
                 'dropoff_longitude','dropoff_latitude','passenger_count']
    )

    assert compare_df(df, df_true)

###############
# Script: src.clean
###############

def test_remove_missing_obs_happy():
    df = make_raw_data()
    df.iloc[1, 1] = None
    df = remove_missing_obs(df)

    df_true = pd.DataFrame([
        ['2010-01-05 16:52:16.0000002',16.9,'2010-01-05 16:52:16 UTC',-74.016048,40.711303,-73.979268,40.782004,1]],
        columns=['key','fare_amount','pickup_datetime','pickup_longitude','pickup_latitude',
                 'dropoff_longitude','dropoff_latitude','passenger_count']
    )

    assert compare_df(df, df_true)

# input is not a df
def test_remove_missing_obs_unhappy():
    try:
        remove_missing_obs([None, 'a', 'b'])
        assert False
    except TypeError:
        assert True

def test_clean_key_happy():
    df = make_raw_data()
    df = clean_key(df)

    df_true = pd.DataFrame(
        [
            [16.9, '2010-01-05 16:52:16 UTC', -74.016048, 40.711303, -73.979268, 40.782004, 1],
            [4.5, '2009-06-15 17:26:21 UTC', -73.844311, 40.721319, -73.84161, 40.712278, 1]
        ],
        columns=['fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude','dropoff_longitude',
                 'dropoff_latitude', 'passenger_count']
    )

    assert compare_df(df, df_true, str_col=None)

# if key does not exist, return original df
def test_clean_key_unhappy():
    df = make_raw_data()
    df.drop(['key'], inplace=True, axis=1)
    df = clean_key(df)

    df_true = make_raw_data()
    df_true.drop(['key'], inplace=True, axis=1)

    assert compare_df(df, df_true, str_col=None)


def test_clean_fare_amount_happy():
    df = make_raw_data()
    df.loc[len(df), :] = [
        '2011-06-13 12:26:12.0000001', 0.5, '2011-06-13 12:26:12 UTC', -73.844311, 40.721319, -73.84161, 40.712278, 1
    ]
    df = clean_fare_amount(df, 2.5)
    df_true = make_raw_data()
    assert compare_df(df, df_true)

# if fare_amount <= 0, raise value error
def test_clean_fare_amount_unhappy():
    try:
        df = make_raw_data()
        clean_fare_amount(df, -1)
        assert False
    except ValueError:
        assert True

def test_clean_locations_happy():
    df = make_raw_data()
    # -100 < nyc_min_lon = -75
    df.loc[len(df), :] = [
        '2011-06-13 12:26:12.0000001', 0.5, '2011-06-13 12:26:12 UTC', -100, 40.721319, -73.84161, 40.712278, 1
    ]
    df = clean_locations(df, nyc_min_lon=-75, nyc_max_lon=-72, nyc_min_lat=39, nyc_max_lat=42)

    df_true = make_raw_data()
    assert compare_df(df, df_true)

# if any longitude and latitude inputs is not numeric, set it to default value
def test_clean_locations_unhappy():
    try:
        df = make_raw_data()
        clean_locations(df, nyc_min_lon='-75', nyc_max_lon='-72', nyc_min_lat='39', nyc_max_lat='42')
        assert False
    except TypeError:
        assert True

def test_clean_passenger_count_happy():
    df = make_raw_data()
    # count = 9 > max
    df.loc[len(df), :] = [
        '2011-06-13 12:26:12.0000001', 30, '2011-06-13 12:26:12 UTC', -100, 40.721319, -73.84161, 40.712278, 9
    ]
    df = clean_passenger_count(df, min_count=1, max_count=5)

    df_true = make_raw_data()
    assert compare_df(df, df_true)

# min_count should be positive and less than max_count
def test_clean_passenger_count_unhappy():
    try:
        df = make_raw_data()
        df = clean_passenger_count(df, min_count=5, max_count=0)
        assert False
    except ValueError:
        assert True

###############
# Script: src.featurize
###############

def test_generate_hour_happy():
    df = format_df(make_clean_data(), str_col=None)
    df = generate_hour(df)

    df_true = format_df(make_clean_data(), str_col=None)
    df_true['pickup_hour'] = [16, 17]
    assert compare_df(df, df_true, str_col=None)

def test_generate_hour_unhappy():
    try:
        generate_hour('df')
        assert False
    except TypeError:
        assert True

def test_generate_dayofweek_happy():
    df = format_df(make_clean_data(), str_col=None)
    df = generate_dayofweek(df)

    df_true = make_clean_data()
    df_true['pickup_dayofweek'] = ['Tuesday', 'Monday']
    assert compare_df(df, df_true, str_col=None)

def test_generate_dayofweek_unhappy():
    try:
        df = make_clean_data()
        df.drop(['pickup_datetime'], axis=1, inplace=True)
        generate_dayofweek(df)
        assert False
    except KeyError:
        assert True

def test_generate_distance_happy():
    df = make_clean_data()
    df = generate_distance(df)

    df_true = make_clean_data()
    df_true["lat_diff"] = np.abs(df.dropoff_latitude - df.pickup_latitude)
    df_true["lon_diff"] = np.abs(df.dropoff_longitude - df.pickup_longitude)
    df_true['distance'] = ((df_true.lat_diff) ** 2 + (df_true.lon_diff) ** 2) ** .5
    df_true.drop(['lat_diff', 'lon_diff'], axis=1, inplace=True)

    assert compare_df(df, df_true, str_col=None)

def test_generate_distance_unhappy():
    try:
        df = make_clean_data()
        df.loc[0,'pickup_longitude'] = '45'
        generate_distance(df)
        assert False
    except TypeError:
        assert True

###############
# Script: src.split
###############

def test_stratified_sampling_happy():
    df = make_features_data()
    train_df, test_df = stratified_sampling(df, strata_cols=['pickup_hour', 'pickup_dayofweek'])
    train_df['strata'] = train_df['pickup_hour'].astype(str) + train_df['pickup_dayofweek'].astype(str)
    test_df['strata'] = test_df['pickup_hour'].astype(str) + test_df['pickup_dayofweek'].astype(str)
    assert train_df['strata'].value_counts().equals(test_df['strata'].value_counts())

# sample observations > # of rows in df
# function will set sample observations = # of rows in df and continue
def test_stratified_sampling_unhappy():
    df = make_features_data()
    train_df, test_df = stratified_sampling(df, strata_cols=['pickup_hour', 'pickup_dayofweek'], sample_obs=50000)
    train_df['strata'] = train_df['pickup_hour'].astype(str) + train_df['pickup_dayofweek'].astype(str)
    test_df['strata'] = test_df['pickup_hour'].astype(str) + test_df['pickup_dayofweek'].astype(str)
    assert train_df['strata'].value_counts().equals(test_df['strata'].value_counts())

def test_one_hot_encoder_happy():
    df = make_features_data()
    df = one_hot_encoder(df, {'pickup_dayofweek': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                                                   'Saturday', 'Sunday']})

    if (df.loc[0:1, 'pickup_dayofweek_Thursday'].all() == 0) and (df.loc[0:1, 'pickup_dayofweek_Sunday'].all() == 1) and \
            (df.loc[2:3, 'pickup_dayofweek_Thursday'].all() == 1) and (df.loc[2:3, 'pickup_dayofweek_Sunday'].all() == 0):
        assert True
    else:
        assert False

# one of the column needs to be one-hot encoded does not exist
def test_one_hot_encoder_unhappy():
    df = make_features_data()
    df.drop(['pickup_dayofweek'], axis=1, inplace=True)

    try:
        one_hot_encoder(df, ['pickup_dayofweek', 'pickup_hour'])
        assert False
    except KeyError:
        assert True

###############
# Script: src.train
###############

def test_train_rf_model_happy():
    df = make_train_data()
    model, imp = train_rf_model(df, target_column='fare_amount')

    # check model type and all features are included in feature importance data frame
    assert isinstance(model, sklearn.ensemble._forest.RandomForestRegressor) and imp.shape[0] == df.shape[1]-1

# if any column specified in feature_columns does not exist, raise key error
def test_train_rf_model_unhappy():
    try:
        df = make_train_data()
        train_rf_model(df, feature_columns=['pickup_longitude', 'not_exist_feature'])
        assert False
    except KeyError:
        assert True

###############
# Script: src.score
###############

def test_score_model_happy():
    df_train = make_train_data()
    model, imp = train_rf_model(df_train, target_column='fare_amount')

    df_test = make_test_data()
    df_pred = score_model(df_test, model)

    assert np.issubdtype(df_pred['ypred_test'].dtype, np.number) and np.issubdtype(df_pred['y_test'].dtype, np.number)

# target column doesn't exist
def test_score_model_unhappy():
    try:
        df_train = make_train_data()
        model,_ = train_rf_model(df_train, target_column='fare_amount')

        df_test = make_test_data()
        score_model(df_test, model, target_column='not_exist')
        assert False
    except KeyError:
        assert True

###############
# Script: src.evaluate
###############

def test_evaluate_model_happy():
    df = make_pred_data()
    mae, mse, rmse, r2 = evaluate_model(df)
    assert all(isinstance(x, Number) for x in [mae, mse, rmse, r2])

def test_evaluate_model_unhappy():
    try:
        df = make_pred_data()
        df.drop(['y_test'], axis=1, inplace=True)
        evaluate_model(df)
        assert False
    except KeyError:
        assert True



