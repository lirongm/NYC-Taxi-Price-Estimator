data/raw_data.csv: config/config.yaml
	python3 run.py download --s3_bucket=nw-lma-s3 --s3_key=data/raw_data.csv --output=data/raw_data.csv
download: data/raw_data.csv

data/filtered-data.csv: data/raw_data.csv config/config.yaml
	python3 run.py filter --config=config/config.yaml --input=data/raw_data.csv --output=data/filtered-data.csv
filter: data/filtered-data.csv

data/clean-data.csv: data/raw_data.csv config/config.yaml
	python3 run.py clean --config=config/config.yaml --input=data/filtered-data.csv --output=data/clean-data.csv
clean: data/clean-data.csv

data/features-data.csv: data/clean-data.csv config/config.yaml
	python3 run.py featurize --config=config/config.yaml --input=data/clean-data.csv --output=data/features-data.csv
featurize: data/features-data.csv

data/train-data.csv data/test-data.csv: data/features-data.csv config/config.yaml
	python3 run.py split --config=config/config.yaml --input=data/features-data.csv --output_train=data/train-data.csv \
						 --output_test=data/test-data.csv
split: data/train-data.csv data/test-data.csv

model/model.pkl evaluation/feature-imp.csv: data/train-data.csv config/config.yaml
	python3 run.py train --config=config/config.yaml --input=data/train-data.csv --output_model=model/model.pkl \
						 --output_feature_imp=evaluation/feature-imp.csv
train: model/model.pkl evaluation/feature-imp.csv

data/test-predictions.csv: data/test-data.csv model/model.pkl config/config.yaml
	python3 run.py score --config=config/config.yaml --input_data=data/test-data.csv --input_model=model/model.pkl \
						 --output=data/test-predictions.csv
score: data/test-predictions.csv

evaluation/test-metrics.txt: data/test-predictions.csv config/config.yaml
	python3 run.py evaluate --config=config/config.yaml --input=data/test-predictions.csv \
							--output=evaluation/test-metrics.txt
evaluate: evaluation/test-metrics.txt

unit_tests:
	pytest unit_tests.py

pipeline: download filter clean featurize split train score evaluate

.PHONY: download filter clean featurize split train score evaluate pipeline unit_tests
