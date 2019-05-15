.PHONY: db

DATA_LINK="https://s3.amazonaws.com/instacart-datasets/instacart_online_grocery_shopping_2017_05_01.tar.gz"

data/external/raw_data.tar.gz:
	curl -X GET ${DATA_LINK} -o data/external/raw_data.tar.gz && tar -xvzf data/external/raw_data.tar.gz -C data/external && mv data/external/instacart_2017_05_01/* data/external && rmdir data/external/instacart_2017_05_01

data: data/external/raw_data.tar.gz

data/external/aisles.csv: data
data/external/departments.csv: data
data/external/order_products.csv: data
data/external/order_products_prior.csv: data
data/external/orders.csv: data
data/external/products.csv: data

s3: data/external/aisles.csv data/external/departments.csv data/external/order_products.csv data/external/order_products_prior.csv data/external/orders.csv data/external/products.csv
	python src/upload_s3.py

data/features/shoppers.csv: data/external/aisles.csv data/external/departments.csv data/external/order_products.csv data/external/order_products_prior.csv data/external/orders.csv data/external/products.csv
	python src/generate_features.py

features: data/features/shoppers.csv data/features/baskets.csv

# can change --mode to 'rds' to use RDS db (must have MYSQL_X vars in env)
db: models/db.py
	python models/db.py --mode local

ingest: models/db.py data/features/baskets.csv
	python models/db.py --mode local --populate True

setup: data s3 features ingest

models/model.pkl: data/features/user-features.csv src/train_model.py config/model_config.yml
	python src/train_model.py

trained-model: models/model.pkl

all: setup trained-model

# Create a virtual environment named instacart-env
instacart-env/bin/activate: requirements.txt
	test -d instacart-env || virtualenv instacart-env
	. instacart-env/bin/activate && pip install -r requirements.txt
	touch instacart-env/bin/activate

venv: instacart-env/bin/activate

# Run the Flask application
app:
	python run.py app

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

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	rm -rf .pytest_cache

clean: clean-tests clean-env clean-pyc

clean-db:
	rm data/instacart.db
