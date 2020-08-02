import os
import pandas as pd
import logging

from src.helpers import load_yaml, write_csv, check_path

logger = logging.getLogger(__name__)

def filter_year(df, year = 2015):
    """ Filter data by a specific year
    Args:
        df (`pandas.DataFrame`): Raw data
        year (int): Filter data by a specific year
    Returns:
        df (`pandas.DataFrame`): Filtered data
    """

    if 'pickup_datetime' not in list(df.columns):
        raise KeyError("Data does not contain a pickup_datetime field. Columns in data are %s" % df.columns.to_list())

    # Convert pickup_datetime to datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format="%Y-%m-%d %H:%M:%S UTC")
    # Get year
    df['pickup_year'] = df['pickup_datetime'].dt.year
    # Drop obs other than the specified year
    df.drop(index=df[df.pickup_year != year].index, inplace=True)
    # drop year column afterwards
    df.drop(['pickup_year'], axis=1, inplace=True)
    return df

def process_by_chunk(file_path, year = 2015, max_num_rows_read = None, chunksize = 10000, log_per_chunks = 500):
    """ Read and filter data by chunks
    Args:
        file_path (`str`): The path to the raw data
        year (int): The specific year where data is filtered by. Optional, default is most recent year in raw data, 2015
        max_num_rows_read: The max number of rows read from raw data. Optional, default is None, which indicates reading
            all data
        chunksize: The chunk size. Optional, default is None when max_num_rows_read is not None, indicating read all
            data at once. Default is 10000 when max_num_rows_read is None, since the data is too big to be read once.
        log_per_chunks: After this number of chunks gets done, log information once. Optional, default = 500 chunks.
    Returns:
        df (`pandas.DataFrame`): Filtered data frame
    """
    if os.path.exists(file_path) is False:
        raise FileNotFoundError("Failed to read and filter data by chunks, since the file path does not exist")

    # create a data frame to save results
    filtered_df = pd.DataFrame()

    # set a counter for the number of chunks that have been filtered
    counter = 0

    # read all data by chunks if max_num_rows_read is None
    if max_num_rows_read is None:

        logger.info("Starting to read raw data by chunk and filter by year = %i" % year)
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # if filtered_df is empty, assign column names to it
            if not list(filtered_df.columns):
                filtered_df = pd.DataFrame(columns=chunk.columns)
            # append valid rows
            filtered_df = filtered_df.append(filter_year(chunk, year))

            # increment counter
            counter = counter + 1

            # log info after each `log_per_chunks` chunks done
            if counter % log_per_chunks == 0:
                logger.info("Filtered data by year for %i chunks" % counter)

        logger.info("Filtered all data by year = %i, and the filtered data has %i rows" % (year, filtered_df.shape[0]))

    # read max_num_rows_read rows of data by chunks if max_num_rows_read is not None
    else:
        # if chunksize > total # of rows needed to read
        # set it to be total # of rows needed to read, so read data at once
        if max_num_rows_read < chunksize:
            logger.warning("Chunksize reset from %i to %i, since it is greater than the total number of rows needed to read"
                           % (chunksize, max_num_rows_read))
            chunksize = max_num_rows_read

        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # if filtered_df is empty, assign column names to it
            if not list(filtered_df.columns):
                filtered_df = pd.DataFrame(columns=chunk.columns)
            # append valid rows
            filtered_df = filtered_df.append(filter_year(chunk, year))

            # increment counterer
            counter = counter + 1

            # log info after each `log_per_chunks` chunks done
            if counter % log_per_chunks == 0:
                logger.info("Filtered data by year for %i chunks" % counter)

            # stop if we reach total # of rows we need to read
            if counter * chunksize == max_num_rows_read:
                break
        logger.info("Filtered %i rows of data by year = %i, and the filtered data has %i rows" %
                    (max_num_rows_read, year, filtered_df.shape[0]))

    return filtered_df

def run_filter(args):
    """ Wrapper function to pass in args, read data, load configuration, and execute filter_year function """
    logger.info("------------------Starting to filter data by year-----------------")
    logger.info("Please note it will take a while (max = ~2h) to run the filter step, as the raw data has ~50 million observations")

    # read configuration
    config = load_yaml(args.config)

    # filter data by chunks
    filtered_df = process_by_chunk(args.input, **config['filter'])

    # save filtered data
    logger.info("Starting to write filtered data")
    write_csv(filtered_df, path=args.output, description='Filtered data')

    logger.info("------------------Finished filtering data-----------------")
