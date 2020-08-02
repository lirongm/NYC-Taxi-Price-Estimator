import pandas as pd
import logging.config

from flask import Flask
from flask import render_template, request
from flask_sqlalchemy import SQLAlchemy

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from src.create_db import Prediction
from src.featurize import generate_distance
from src.helpers import load_model
from src.split import one_hot_encoder

# Initialize the Flask application
app = Flask('NYC_Taxi_Fare', template_folder="app/templates", static_folder="app/static")

# Configure flask app from flask_config.py
app.config.from_pyfile('config/flaskconfig.py')

# Define LOGGING_CONFIG in flask_config.py - path to config file for setting
# up the logger (e.g. config/logging/logging.conf)
logging.config.fileConfig(app.config["LOGGING_CONFIG"], disable_existing_loggers=False)
logger = logging.getLogger(app.config["APP_NAME"])
logger.debug('Test log')

# Initialize the database
db = SQLAlchemy(app)


@app.route('/')
def index():
    """Main view that contains NYC Taxi price estimator and instructions

    Create view into index page using template app/templates/index.html
    Users can browse instructions and enter required inputs for prediction

    Returns: rendered html template

    """

    try:
        return render_template('index.html')
    except:
        logger.warning("Not able to display homepage (index.html), error page returned")
        return render_template('error.html')


def geocoding(address, description=None):
    """Convert address to longitude and latitude using Nominatim Geocoding service and return longitude and latitude
    
    Args:
        address (`str`): address needs to converted to longitude and latitude
        description (`str`): the specific name for the address used for logging information
    
    Returns:
        latitude (`float`): latitude for the address
        longitude (`float`): longitude for the address
    
    """
    # address has to be string
    if isinstance(address, str) is False:
        raise TypeError("address must be a string")

    # use Nominatim Geocoding service, which is built on top of OpenStreetMap data
    locator = Nominatim(user_agent='Geocoder')
    # delay between geocoding calls
    geocode = RateLimiter(locator.geocode, min_delay_seconds=app.config['MIN_DEALY_SECONDS'])

    # get location object from address
    # it will return None if it cannot find the address
    location = geocode(address)
    # check invalid address and return None if invalid
    if location is None:
        logger.error('The %s address is invalid. Cannot find it in Nominatim Geocoding service.' % description)
        return None, None

    # extract latitude and longitude
    latitude, longitude, _ = tuple(location.point)
    logger.info("Latitude and longitude for %s have been extracted" % description)

    return latitude, longitude


@app.route('/predict', methods=['POST'])
def predict():
    """View that process a POST with new user inputs
    Returns: rendered html template with prediction.
    """
    try:
        #####################
        # get user input
        #####################
        pickup_address = request.form['pickup_address']
        dropoff_address = request.form['dropoff_address']
        passenger_count = request.form['passenger_count']
        pickup_date = request.form['pickup_date']
        pickup_time = request.form['pickup_time']
        logger.info("All user inputs have been retrieved")

        #####################
        # extract features from user input
        #####################
        # extract pickup and dropoff latitude and longitutde
        pickup_latitude, pickup_longitude = geocoding(pickup_address, description="pickup")
        if pickup_latitude is None:
            return render_template('pickup_address_error.html')

        dropoff_latitude, dropoff_longitude = geocoding(dropoff_address, description="dropoff")
        if dropoff_latitude is None:
            return render_template('dropoff_address_error.html')

        #  create a data frame to save features for prediction
        df = pd.DataFrame([[pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count]],
                          columns=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                                   'passenger_count'])

        # generate distance
        df = generate_distance(df)
        logger.info("distance has been extracted")

        # generate pickup_hour
        try:
            pickup_date = pd.to_datetime(pickup_date, infer_datetime_format=True)
            pickup_dayofweek = pickup_date.day_name()
            df.loc[0, 'pickup_dayofweek'] = pickup_dayofweek
            logger.info("pickup_dayofweek has been extracted")
        except:
            return render_template('pickup_date_error.html')

        # generate pickup_dayofweek
        try:
            pickup_time = pd.to_datetime(pickup_time, infer_datetime_format=True)
            pickup_hour = pickup_time.hour
            df.loc[0, 'pickup_hour'] = pickup_hour
            # ensure pickup_hour column is integer to be consistent with training set and database
            df['pickup_hour'] = df['pickup_hour'].astype(int)
            logger.info("pickup_hour has been extracted")
        except:
            return render_template('pickup_hour_error.html')


        # one hot encode features specified in configurations
        one_hot_dict = app.config['ONE_HOT_ENCODER']
        df = one_hot_encoder(df, one_hot_dict)

        logger.info("All features have been extracted and transformed")

        #####################
        # check whether the same user inputs exist in database:
        # if exists, get prediction from database
        # if not, load model, make prediction, and add record to database
        #####################

        try:
            record = db.session.query(Prediction).filter_by(
                pickup_longitude=float(df.loc[0, 'pickup_longitude']),
                pickup_latitude=float(df.loc[0, 'pickup_latitude']),
                dropoff_longitude=float(df.loc[0, 'dropoff_longitude']),
                dropoff_latitude=float(df.loc[0, 'dropoff_latitude']),
                passenger_count=int(df.loc[0, 'passenger_count']),
                pickup_hour=int(pickup_hour),
                pickup_dayofweek=str(pickup_dayofweek),
            ).first()
        except Exception as e:
            logger.error("Failed to query database, since %s" % e)

        if record is not None:
            prediction = record.predicted_fare
            logger.info("User inputs exist in database and prediction has been extracted from database. The predicted "
                        "fare is %.2f" % prediction)

        else:
            logger.info("User inputs do not exist in database. Loading model to make prediction.")

            # load model
            model = load_model(app.config['MODEL_PATH'])

            # make prediction and add it to data frame
            try:
                prediction = model.predict(df)
            except Exception as e:
                logger.error("Failed to make prediction, since %s " % e)


            # prediction is numpy.ndarray, so we get the first element
            prediction = prediction[0]
            df.loc[0, 'predicted_fare'] = prediction
            logger.info("The prediction for fare amount has been made: the predicted fare is %.2f" % prediction)

            try:
                # add records to database
                prediction1 = Prediction(
                    pickup_longitude=float(df.loc[0, 'pickup_longitude']),
                    pickup_latitude=float(df.loc[0, 'pickup_latitude']),
                    dropoff_longitude=float(df.loc[0, 'dropoff_longitude']),
                    dropoff_latitude=float(df.loc[0, 'dropoff_latitude']),
                    passenger_count=int(df.loc[0, 'passenger_count']),
                    pickup_hour=int(pickup_hour),
                    pickup_dayofweek=str(pickup_dayofweek),
                    predicted_fare=float(df.loc[0, 'predicted_fare'])
                )
                db.session.add(prediction1)
                db.session.commit()
                logger.info("New prediction record has been added.")
            except Exception as e:
                logger.error("Failed to add the record to database, since %s" % e)
        return render_template('index.html', result=round(prediction, 2))

    except:
        logger.warning("Not able to make prediction, error page returned")
        return render_template('error.html')


if __name__ == '__main__':
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"], host=app.config["HOST"])
