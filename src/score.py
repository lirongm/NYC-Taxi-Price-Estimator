import sys
import pickle
import logging
import pandas as pd

from src.helpers import load_yaml, read_csv, write_csv, load_model

logger = logging.getLogger(__name__)

def score_model(data, model, feature_columns=None, target_column='fare_amount'):
    """ Generate prediction on test set
    Args:
        data (`pandas.DataFrame`): The test set data frame.
        model (`sklearn.linear_model.LogisticRegression`): The trained model object.
        feature_columns (:obj:`list` of :obj:`str`): List of feature column names. If not provided, then every columns
            except the target column and `test` indicator column will be used as features.
        target_column (`str`): Column name of the target. If not provided, 'fare_amount' will be used as default.
    Returns:
        df (`pandas.DataFrame`): The data frame which includes true target (`y_test`) and prediction (`ypred_test`)
            on test set.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("The `data` input has to be pd.DataFrame")

    if target_column not in list(data.columns):
        raise KeyError("Failed to score model: the target column does not exist in data frame")

    # create a data frame to save the predictions
    df = pd.DataFrame(columns=['y_test', 'ypred_test'])

    df['y_test'] = data.loc[:, target_column]


    # get features from test set
    # if feature columns are not specified, use all columns other than the target column as features
    if feature_columns is None:
        X_test = data.loc[:, ~data.columns.isin([target_column])]
    else:
        X_test = data.loc[:, data.columns.isin(feature_columns)]
        if X_test.shape[1] != len(feature_columns):
            raise KeyError("At least one column in feature_columns does not exist in data frame")


    # get predictions
    ypred_test = model.predict(X_test)

    # add predictions to data frame
    df['ypred_test'] = ypred_test

    return df


def run_score(args):
    """Load configuration file and pass argparse args which include args.input_model, args.input_data,
    args.output, and args.config """

    logger.info("-------------Starting to score model-------------")

    # load configuration
    config = load_yaml(args.config)
    # read data
    data = read_csv(args.input_data)
    # load model
    model = load_model(args.input_model)

    output = score_model(data, model, **config['score'])
    write_csv(output, args.output, description="Predictions on test set")

    logger.info("-------------Finished scoring model-------------")
