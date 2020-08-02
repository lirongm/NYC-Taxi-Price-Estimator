  
# New York City Taxi Price Estimator Project Repository  
Created by Lirong Ma, QA by Yanmeng (Selina) Song   
  
<!-- toc -->  
- [Project Charter](#project-charter)  
- [Backlog](#backlog)  
  
- [Directory Structure](#directory-structure)  
- [Download raw data from Kaggle](#download-raw-data-from-kaggle)
- [Running the Model Training Pipeline](#running-the-model-training-pipeline)  
	*   [Set up AWS credentials ](#set-up-aws-credentials)  
	*   [Configure artifact outputs ](#configure-artifact-outputs)
	*   [Build docker image ](#build-docker-image-1)  
	*   [Run the pipeline ](#run-the-pipeline)  
	* 	[Run unit tests ](#run-unit-tests)  
- [Running the App](#running-the-app)  
  * [Set up MySQL Credentials ](#set-up-mysql-credentials)  
  * [Build docker image](#build-docker-image-2)  
  * [Run the app using RDS](#run-app-local-rds)  
  * [Run the app using a local database](#run-app-local-db)  
  
<!-- tocstop -->  
  
## Project Charter  
### Vision  
This project aims to predict the fare amount of a taxi ride in New York City (NYC) given the pickup time, the pickup and dropoff locations, and the number of passengers. Ride-hailing applications such as Uber and Lyft provide price estimates, which makes it possible for customers to decide which application they want to use for a certain trip ahead of time, whereas we often do not know how much it costs for a certain taxi ride until the end of the trip.  In order to address this problem, this application will provide the estimated fare of a taxi ride in NYC for users, so they are able to compare it with price estimates provided by ride-hailing applications to plan their next trip, and there are no surprises at the end of the  ride.  
  
### Mission  
The application will ask users to provide information (e.g. pickup time, pickup and dropoff locations, and number of passengers) through a user interface. A supervised model will be trained off-line on the training dataset, which contains approximately 55 million past taxi rides with 6 features [[https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data)]. The application will expose the predicted taxi fare made by the trained model based on user inputs.  
  
### Success Criteria  
Two major success criteria will be considered:  
1. The machine learning performance metric:  
The Root Mean Square Error (RMSE) has to be less than $4 for the application to go live. One benefit of RMSE is that the error is intuitive, as it is given in the units being measured, so we can know directly how incorrect the model might be on new data. $4 is chosen as the threshold, because a basic taxi fare estimate based solely on distance between pickup location and destination results in an RMSE of $5-$8. The application has to provide lower RMSE than basic estimates to be useful.  
  
2. The business success metric:  
The application will be considered as successful, if it is used to estimate 100 trips per day, approximately 0.01% of total trips per day by ride-hailing applications and taxi in NYC. Retention rate would also be a useful metric, but it is hard to be assessed, as we do not require users to log in to use the application.  
  
## Planning  
### Initiative 1: Develop supervised machine learning models to predict NYC taxi fare and find the best model with lowest RMSE  
* Epic 1: Prepare for development  
   * Story 1: Define the problem and evaluation criteria  
   * Story 2: Assess modeling and deployment requirements  
   * Story 3: Identify source datasets, assess data access, and address gaps in the datasets  
* Epic 2: Iterate on model  
   * Story 1: Assess data quality and perform exploratory data analysis  
   * Story 2: Perform data cleaning, including removing duplicate or irrelevant observations, fixing structural errors, filtering unwanted outliers, and handling missing data.  
   * Story 3: Compile and generate features  
   * Story 4: Perform sanity checks on all the data to ensure they are accurate and consistent before building models  
   * Story 5: Perform feature selection  
   * Story 6: Develop, tune, and evaluate different types of supervised machine learning models using cross validations  
   * Story 7: Choose the best model, quantify uncertainty of model results, summarize analyses, and get feedback on approach  
  
* Epic 3: Prepare for production deployment  
   * Story 1: Migrate code to scripts  
   * Story 2: Assess for missing unit tests and logging to ensure handling unexpected input properly  
   * Story 3: Review and test requirements.txt  
   * Story 4: Go through code reviews  
   * Story 5: Create plan for model performance monitoring  
  
### Initiative 2: Develop an interactive web application that asks users to provide taxi ride information and then exposes the predicted fare amount  
* Epic 1: Identify deployment artifacts  
   * Story 1: Establish model execution and training pipeline artifacts  
   * Story 2: Set operational configurations  
  
* Epic 2: Choose deployment strategy and configure target environment  
   * Story 1: Identify target environment and configure deployment automation  
   * Story 2: Update security configurations  
  
* Epic 3: Design user interface for interactions  
  
* Epic 4: Use S3 bucket to store raw source data and create table in database for model result  
  
* Epic 5: Create web application (Flask) frontend that can be deployed locally on a computer using Docker that exposes the results of the model through HTML and CSS  
  
* Epic 6: Deploy artifacts  
   * Story 1: Release to initial subsets of users  
   * Story 2: Fix technical issues and validate results of treatment group  
   * Story 3: Release to full population  
  
## Backlog  
  
The stories are assigned with points to represent the time estimation to complete them:  
- 0 points - quick chore  
- 1 point ~ 1 hour  
- 2 points ~ 1/2 day  
- 4 points ~ 1 day  
  
1. Initiative1.epic1.story1 (2 points)  
2. Initiative1.epic1.story2 (2 points)  
3. Initiative1.epic1.story3 (2 points)  
4. Initiative1.epic2.story1 (4 points)  
5. Initiative1.epic2.story2 (4 points)  
6. Initiative1.epic2.story3 (4 points)  
7. Initiative1.epic2.story4 (1 points)  
8. Initiative1.epic2.story5 (2 points)  
9. Initiative1.epic2.story6 (4 points)  
10. Initiative1.epic2.story7 (2 points)  
11. Initiative1.epic3.story1 (4 points)  
12. Initiative1.epic3.story2 (2 points)  
13. Initiative1.epic3.story3 (2 points)  
14. Initiative1.epic3.story4 (2 points)  
  
## Icebox  
- Initiative1.epic3.story5  
- Initiative2.epic1.story1  
- Initiative2.epic1.story2  
- Initiative2.epic2.story1  
- Initiative2.epic2.story2  
- Initiative2.epic3  
- Initiative2.epic4  
- Initiative2.epic5  
- Initiative2.epic6.story1  
- Initiative2.epic6.story2  
- Initiative2.epic6.story3  
  
  
## Directory Structure  
  
```  
├── README.md                         <- You are here  
├── app  
│   ├── static/                       <- CSS, JS files that remain static  
│   ├── templates/                    <- HTML (or other code) that is templated and changes based on a set of inputs  
│   ├── boot.sh                       <- Start up script for launching app in Docker container  
│   ├── Dockerfile                    <- Dockerfile for building image to run app │  
├── config                            <- Directory for configuration files  
│   ├── local/                        <- Directory for keeping environment variables and other local configurations that *do not sync** to Github  
│   ├── logging/                      <- Configuration of python loggers  
│   ├── .awsconfig                    <- Configuration of AWS credentials  
│   ├── .mysqlconfig                  <- Configuration of MySQL database  
│   ├── config.yaml                   <- Configuration for Python scripts  
│   ├── flaskconfig.py                <- Configurations for Flask API 
|
├── deliverables/                     <- White papers, presentations, final work products that are presented or delivered  
│  ├── Presentation.pdf               <- Final presentation slides  
|  
├── src/                              <- Source code for the project  
│  ├── s3_upload.py                   <- Upload raw data to S3 bucket  
│  ├── create_db.py                   <- Create database locally or in RDS  
│  ├── download.py                    <- Download raw data from S3  
│  ├── filter.py                      <- Read a part or all raw data and filter by a specific year  
│  ├── clean.py                       <- Clean data  
│  ├── featurize.py                   <- Feature engineering  
│  ├── split.py                       <- Perform stratified samplings to generate training and test sets and one-hot-encoding categorical variables  
│  ├── train.py                       <- Train a Random Forest Regressor on the training set  
│  ├── score.py                       <- Predict on the test set  
│  ├── evaluate.py                    <- Calculate evaluation metrics on the test set  
│  ├── helpers.py                     <- Helper functions to read and write files  
│  ├── unit_tests_helpers.py          <- Helper functions to make dataframe and format dataframes for comparison for unit tests  
|  
├── run.py                            <- Simplifies the execution of one or more of the src scripts 
├── app.py                            <- Flask wrapper for running the model  
├── unit_tests.py                     <- Unit tests for each applicable function in source code  
├── Dockerfile                        <- Dockerfile for building image to run model pipeline  
├── Makefile                          <- Makefile for running model pipeline  
├── requirements.txt                  <- Python package dependencies  
```  

## Download raw data from Kaggle 

-  Please go to [https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data?select=train.csv](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data?select=train.csv)
-  To download the dataset, you have to sign up for a Kaggle account. If you have an account, please `sign in`. If you do not have one, please `register` and then sign in. 
 - On the webpage, click `I understand and agree` and then `download` to download the dataset **train.csv**. *Note*: if you do not sign in, the webpage will indicate "an error occurred accepting the competition rules."
- Rename **train.csv** as **raw_data.csv** and move it to NYC-Taxi-Price-Estimator/data.

## Running the Model Training Pipeline 

### Set up AWS credentials
Fill in AWS credentials and export them to prepare for downloading raw data from S3 bucket. 
```bash
export AWS_ACCESS_KEY_ID=  
export AWS_SECRET_ACCESS_KEY=
```

### Configure artifact outputs
S3 bucket name and key (file path in S3) and all the input and out file paths are configurable through command line arguments in Makefile. They all have default values and please look up help for each argument to confirm what it is for. Please feel free to change them to any path you desire. The outputs from each step are going to be used in the subsequent steps, so please ensure to change all the corresponding ones if you make any changes. 

### Build docker image 
Please go to the root repository and build the docker image.
```bash  
docker build -t project .  
```

### Run the pipeline
The raw data contains ~50 million rows of data, so it takes ~15 mins to download the data from S3. If you would like to run the pipeline on a part of the raw data or reduce the training and test sets, please adjust the following configurations in config.ymal: 

- `max_num_rows_read`: the maximum number of rows read from the raw data to perform following steps. Default: 1,000,000.
- `chunksize`: the number of rows read once, as the raw data will be read by chunks. Default: 10,000.
- `log_per_chunks`: log how many chunks have been done every this number of chunks. Default: 10.
- `sample_obs`: the total number of observations in training and test set sizes. Default: 10,000.
  

```bash  
docker run \  
--mount type=bind,source="$(pwd)",target=/app/ \  
--env AWS_ACCESS_KEY_ID \  
--env AWS_SECRET_ACCESS_KEY \
project pipeline  
```

### Run unit tests
* `unit_tests.py` is the unit tests file.
* Each applicable function in source code will be tested for a happy path and an unhappy path.
```bash  
docker run project unit_tests
```
## Running the App

### Build docker image 
Please go to the root repository and build the docker image.
```bash
docker build -f app/Dockerfile -t nyctaxi .
```

### Run the app using RDS
* Please configure `SQLALCHEMY_DATABASE_URI` and then run the app
```bash
export SQLALCHEMY_DATABASE_URI=
```
```bash  
docker run \  
--env SQLALCHEMY_DATABASE_URI \  
-p 5000:5000 --name test \  
nyctaxi app.py 
```
* Kill and remove the container once finished
```bash
docker kill test
docker rm test  
```

### Run the app using a local database
We first need to create a local database, and then use it to read and write data when running the app. Please export the environment variable `SQLALCHEMY_DATABASE_URI` to specify the path to the local database and then create the database. If this environment variable is not found, a local database will be created at `sqlite:///data/prediction.db`. We use mount in case we want to check our database locally.

* Create a local database
```bash
export SQLALCHEMY_DATABASE_URI=<sqlite:///data/prediction.db or your preferred path>  
```
```bash
docker run \
--mount type=bind,source="$(pwd)",target=/app/ \
--env SQLALCHEMY_DATABASE_URI \
nyctaxi run.py create_local_db
```
* Run the app
```bash  
docker run \  
--mount type=bind,source="$(pwd)",target=/app/ \  
--env SQLALCHEMY_DATABASE_URI \  
-p 5000:5000 --name test \  
nyctaxi app.py 
```
* Kill and remove the container once finished
```bash
docker kill test
docker rm test  
```
