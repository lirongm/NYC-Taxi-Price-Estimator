import csv
import logging
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.helpers import read_csv, load_yaml, check_path

logger = logging.getLogger(__name__)

def evaluate_model(y_data, save_to=None, y_test_name='y_test', ypred_name='ypred_test'):
    """
       Evaluate model performance on test dataset and generate corresponding reports
    Args:
        y_data (`pandas.DataFrame`): The data frame which includes predicted probabilities, predicted class, and
            true class for each observation in test set.
        save_score_to (`str`): The path to save metrics to. If not given, it will not be saved.
        y_test_name (`str`): The true target column name. Default: `y_test`.
        ypred_name (`str`): The prediction column name. Default: `ypred_test`.
    Returns:
        None
    """
    if not isinstance(y_data, pd.DataFrame):
        raise TypeError("The `df` input has to be pd.DataFrame")

    if (y_test_name or ypred_name) not in list(y_data.columns):
        raise KeyError("At least one required column does not exist in data frame")

    y_test = y_data[y_test_name]
    ypred_test = y_data[ypred_name]

    mae = mean_absolute_error(y_test, ypred_test)
    mse = mean_squared_error(y_test, ypred_test)
    rmse = mean_squared_error(y_test, ypred_test, squared=False)
    r2 = r2_score(y_test, ypred_test)

    # save to a txt if a path is specified
    if save_to is not None:
        check_path(save_to)
        try:
            with open(save_to, 'w') as text_file:
                writer = csv.writer(text_file)
                writer.writerow(["Mean absolute error on test set: "+"{0:.3f}".format(mae)])
                writer.writerow(["Mean squared error on test set: " + "{0:.3f}".format(mse)])
                writer.writerow(["Root mean squared error on test set: " + "{0:.3f}".format(rmse)])
                writer.writerow(["R-squared on test set: " + "{0:.3f}".format(r2)])
                logger.info("MAE, MSE, RMSE, R-squared on test set saved to %s", save_to)
        except Exception as e:
            logger.error(e)
    return mae, mse, rmse, r2

def run_evaluate(args):
    """Load configuration file and pass argparse args which include args.input, args.output, and args.config """

    logger.info("-------------Starting to evaluate model-------------")

    # load configuration
    config = load_yaml(args.config)
    # read data
    data = read_csv(args.input)
    evaluate_model(data, save_to=args.output, **config['evaluate'])
    logger.info("-------------Finished evaluating model-------------")

