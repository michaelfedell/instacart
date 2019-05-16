.PHONY: db

DATA_LINK="https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz"
# Overwrite this by running `make BUCKET="my-test-bucket" s3` for example
BUCKET="instacart-store"
# MODE can be "local" or "rds"
# Overwrite this by running `make MODE="rds" db` for example
MODE="local"

data/external/raw_data.tar.gz:
	curl -X GET ${DATA_LINK} -o data/external/raw_data.tar.gz && tar -xvzf data/external/raw_data.tar.gz -C data/external && mv data/external/instacart_2017_05_01/* data/external && rm -rf data/external/instacart_2017_05_01

data: data/external/raw_data.tar.gz

data/external/aisles.csv: data
data/external/departments.csv: data
data/external/order_products.csv: data
data/external/order_products_prior.csv: data
data/external/orders.csv: data
data/external/products.csv: data

s3: data/external/aisles.csv data/external/departments.csv data/external/order_products.csv data/external/order_products_prior.csv data/external/orders.csv data/external/products.csv
	python src/upload_s3.py --bucket ${BUCKET}

data/features/shoppers.csv: data/external/aisles.csv data/external/departments.csv data/external/order_products.csv data/external/order_products_prior.csv data/external/orders.csv data/external/products.csv
	python src/generate_features.py

features: data/features/shoppers.csv data/features/baskets.csv

# can change MODE to 'rds' to use RDS db (must have MYSQL_X vars in env)
db: src/db.py
	python src/db.py --mode ${MODE}

ingest: src/db.py data/features/baskets.csv
	python modsrcels/db.py --mode ${MODE} --populate

setup: data s3 features ingest

models/model.pkl: data/features/user-features.csv src/train_model.py config/model_config.yml
	python src/train_model.py

trained-model: models/model.pkl

all: setup trained-model

# Create a virtual environment named instacart-env
instacart-env/bin/activate: requirements.txt
	(test -d instacart-env || virtualenv instacart-env) && . instacart-env/bin/activate && pip install -r requirements.txt

	touch instacart-env/bin/activate

venv: instacart-env/bin/activate

.conda-env: requirements.txt
	conda create -n instacart python=3.7 && touch .conda-env

conda: .conda-env
	conda activate instacart && pip install -r requirements.txt

# Run all tests
test:
	py.test

# Clean up things
clean-tests:
	rm -rf .pytest_cache
	rm -r test/model/test/
	mkdir test/model/test
	touch test/model/test/.gitkeep

clean-env:
	rm -r instacart-env
	rm .conda-env

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	rm -rf .pytest_cache

clean: clean-tests clean-env clean-pyc

clean-db:
	rm data/instacart.db
