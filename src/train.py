import sys
import csv
import pickle
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.helpers import check_path, load_yaml, read_csv, write_csv

logger = logging.getLogger(__name__)

def train_rf_model(data, save_model_to=None, save_feature_imp_to=None, feature_columns=None,
                   target_column='fare_amount', random_state=678, **kwargs):
    """Train a logistic regression model and save the model
    Args:
        data (`pandas.DataFrame`): The training set data frame.
        save_model_to (`str`): The path to save the trained model. If not given, it will not be saved.
        save_model_to (`str`): The path to save feature importance. If not given, it will not be saved.
        feature_columns (:obj:`list` of :obj:`str`): List of feature column names. If not provided, then every columns
            except the target column will be used as features.
        target_column (`str`): Column name of the target. If not provided, 'fare_amount' will be used as default.
        random_state (`int`): Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data. Default is 678.
            It will not be used in other solvers such as the default solver ’lbfgs’.
        **kwargs: Keyword arguments for sklearn.ensemble.RandomForestRegressor. Please see sklearn documentation
            for all possible options:
            https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    Returns:
        model (`sklearn.ensemble.RandomForestRegressor`): The trained model object.
        imp_df (`pandas.DataFrame`): The feature importance data frame.
    """

    # get y_train
    if target_column not in list(data.columns):
        raise KeyError("Failed to start model training: the target column does not exist in data frame")

    y_train = data.loc[:, target_column]

    # get features
    # if feature columns are not specified, use all columns other than the target column as features
    if feature_columns is None:
        X_train = data.loc[:, ~data.columns.isin([target_column])]
    else:
        X_train = data.loc[:, data.columns.isin(feature_columns)]
        if X_train.shape[1] != len(feature_columns):
            raise KeyError("At least one column in feature_columns does not exist in data frame")

    # for reproducibility, we specify random_state ahead of time in case users forget to set it in yaml file
    model = RandomForestRegressor(random_state=random_state, **kwargs)
    model.fit(X_train, y_train)

    # get feature importance from model
    featureimp = model.feature_importances_.tolist()

    # create a data frame to save feature importance
    imp_df = pd.DataFrame({"features": list(X_train.columns), "importance": featureimp})
    imp_df.sort_values('importance', ascending=False, inplace=True)

    if save_model_to is not None:
        # make sure the path is valid
        check_path(save_model_to)
        try:
            with open(save_model_to, "wb") as f:
                pickle.dump(model, f)
                logger.info("Trained model object saved to %s", save_model_to)
        except FileNotFoundError:
            logger.error("%s is valid. Please provide a valid path to save the model." % save_model_to)
        except Exception as e:
            logger.error(e)

    if save_feature_imp_to is not None:
        write_csv(imp_df, save_feature_imp_to, description="Feature importance")

    return model, imp_df


def run_train(args):
    """Load configuration file and pass argparse args which include args.input, args.output, and args.config """

    logger.info("-------------Starting to train model-------------")
    config = load_yaml(args.config)
    data = read_csv(args.input)
    train_rf_model(data, args.output_model, args.output_feature_imp, **config['train'])
    logger.info("-------------Finished model training-------------")
