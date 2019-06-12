Built by [Michael Fedell](https://github.com/michaelfedell), QA'd by [Finn Qiao](https://github.com/finnqiao)

# InstaCart

InstaCart has released rich historical data on the grocery shopping habits of their customers. We will use this data to create profiles with predictive power which will help users discover products and stores manage planning and logistical challenges.

Check out the [Project Charter](CHARTER.md) for some background on this project's inception.

Or, to see the planned work, check out the TODO: [issues]() or [ZenHub Board]()

## Project Structure

```txt
├── README.md                         <- You are here
│
├── app
│   ├── static/                       <- CSS, JS files that remain static
│   ├── templates/                    <- HTML (or other code) that is templated and changes based on a set of inputs
│   ├── models.py                     <- Creates the data model for the database connected to the Flask app
│   ├── __init__.py                   <- Initializes the Flask app and database connection
│
├── config/                           <- Directory for yaml configuration files for model training, scoring, etc
│   ├── logging/                      <- Configuration files for python loggers
│   ├── features_config.yml           <- Settings for feature generation

├── data                              <- Folder that contains data used or generated. Only the external/ and sample/ subdirectories are tracked by git
│   ├── archive/                      <- Place to put archive data is no longer used. Not synced with git
│   ├── external/                     <- External data sources, will be synced with git
│   │   ├── cats.yml                  <- Curated list of high-level categories by which to classify grocery aisles
│   │   ├── data_description.md       <- Description of data files provided by Instacart
│   │
│   ├── sample/                       <- Sample data used for code development and testing, will be synced with git
│
├── docs                              <- A default Sphinx project; see sphinx-doc.org for details
│
├── figures/                          <- Generated graphics and figures to be used in reporting
│
├── models/                           <- Trained model objects (TMOs), model predictions, and/or model summaries
│   ├── archive                       <- No longer current models. This directory is included in the .gitignore and is not tracked by git
│
├── notebooks/
│   ├── develop/                      <- Current notebooks being used in development.
│   ├── deliver/                      <- Notebooks shared with others.
│   ├── archive/                      <- Develop notebooks no longer being used.
│   ├── template.ipynb                <- Template notebook for analysis with useful imports and helper functions.
│
├── src                               <- Source scripts for the project
│   ├── archive/                      <- No longer current scripts.
│   ├── helpers/                      <- Helper scripts used in main src files
│   ├── sql/                          <- SQL source code
│   ├── db.py                         <- Script for creating and optionally populating database
│   ├── evaluate_model.py             <- Script for evaluating model performance
│   ├── generate_features.py          <- Script for cleaning and transforming data and generating features used in training and scoring.
│   ├── postprocess.py                <- Script for postprocessing predictions and model results
│   ├── train_model.py                <- Script for training machine learning model(s)
│   ├── upload_s3.py                  <- Script for uploading local files to an S3 bucket
│   ├── score_model.py                <- Script for scoring new predictions using a trained model
│
├── test/                             <- Files necessary for running model tests (see documentation below)

├── run.py                            <- Simplifies the execution of one or more of the src scripts
├── app.py                            <- Flask wrapper for running the model
├── config.py                         <- Configuration file for Flask app
├── requirements.txt                  <- Python package dependencies
```

## Documentation

### Project overview

At a high level, this application takes order and product data and builds a set of order-level features based on temporal stats, basket composition, and other metadata. These orders are then clustered to produce "order_type" labels. Additionally, user profiles are built based on their order histories. A classification model is then trained to predict the order type of a user's next purchase. This model relies on some ~52 attributes mined from order history. To simplify the user interface of the application, these 52 features are mapped to 4 factors via Factor Analysis. Though the model can adapt to changes in feature set, number of order_types, model parameters, etc. The description of clusters (order_types) and feature-factor maps requires manual intervention. The factors can be mapped by examining the `factor_map.png` produced in `data/features` by running the `generate_features.py` script. And the clusters can be examined with help of the `heatmap.png` file which is saved to `app/static` since it is used in the application itself. To facilitate the naming of clusters, `src/name_clusters.py` (or `$ make descriptions`) will connect to the database and add descriptions to ordertypes table based on command line input or a `cluster_desc.csv` file. This is described in further detail in respective scripts.

### Getting Started

Although you should be able to run this project in development without any fuss, a few configurations are required in order to interface with production resources.

Data can be optionally uploaded to/downloaded from an S3 bucket. This will require you to have installed and configured the AWS CLI tools. [More information can be found here](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html)

Additionally, the application can interface with a cloud database instead of a local, SQLite database. This will also require that you have a valid AWS account and a configured RDS instance, with environment variables set for MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, and MYSQL_PORT.

In this phase of the project, all raw data exist in CSV's as downloaded from Instacart [as linked below](##DataLinks). In order to get up and running yourself, you will need to download these large files into `./data/external/` along with several other setup steps required before running the application.

To summarize, the following steps should be taken:

1. Set environment and config files
2. Download the raw data `make data`
3. Generate features from data `make features`
4. Create the database and seed with feature data `make ingest`
5. Train and save the classification model `make trained-model`
6. Run the application `make app`

Alternatively, the required files can be created elsewhere and then downloaded from S3 to run the application

1. Set env variables
2. `make DOWNLOAD=True features`
3. `make ingest`
4. `make DOWNLOAD=True trained-model`
5. `make app`

### Environment

The `MODE` environment variable will control the use of database (should be 'local' or 'rds')  
The `BUCKET` environment variable will point S3 interactions to bucket of that name (default 'instacart-store')  
All `MYSQL_XXX` variables described above will need to be set for rds connection
The `DOWNLOAD` environment variable can be set to "True" or "False" to omit feature gen and model training and instead download needed files from S3 (so that compute-intensive processes can be run separate from application server)

Running `train_model.py` will set a `TMO_PATH` variable to the created model, alternatively, `src.helpers.get_newest_model` can be used to in conjunction with `src.helpers.get_files` to get the created model (as loaded object)

Some of these require override if using `make` commands (see makefile/argparser help)

### Using the Makefile

Makefile directives can be executed by running `make directive`, or `make VAR=X directive` if you want to set environment variable `VAR` as `X` before executing a directive. Examples will follow:

Before beginning work on this project, it is recommended that you create a virtual **environment** with the required packages. Depending on your preferences, this can be done via `virtualenv` or `conda`  
*Note:* if `make conda` fails for you, you may have to run `conda activate instacart && pip install -r requirements.txt`

```bash
make conda
make venv
```

Perform the entire **setup** process from downloading raw data to feature engineering to persistence in S3/database

```bash
make setup
```

Continue the process **all** the way through the modeling and stage at which point the application is ready to run

```bash
make all
```

**Download** raw data from Instacart website and unpack in the appropriate location

```bash
make data
```

**Upload** raw data files (`data/external/*.csv`) to S3 bucket  
*Note:* alternatively, `python src/upload_s3.py --bucket <bucket-name> --dir <local-dir> --file <local-file>` will upload any files matching `local-file` pattern within `local-dir` to S3 in the specified bucket

```bash
make s3
make BUCKET="cool-s3-bucket" s3
```

**Generate features** from the raw data for later use in model development  
*Warning:* this can be a compute-heavy process and may not run well (or at all) on limited resources. feature generation involves clustering on a large dataset and takes about 10 minutes to run on my MacBook with 2.9GHz i7 and 16GB RAM

```bash
make features
```

**Create database** to persist basket (order type) data  
*Note:* the `rds` mode will only work if valid MYSQL config is available in the environment variables (e.g. MYSQL_{USER, PASSWORD, HOST, PORT})

```bash
make db
make MODE="rds" db
```

**Ingest** the created feature data (baskets.csv)  
*Note:* this will also create the table if it does not yet exist

```bash
make ingest
make MODE="rds" ingest
```

## Testing

Unit Tests are implemented for helper/utility functions around the modeling pipeline wherever deemed appropriate. To run tests, simply execute `$ pytest` from the project root directory


## Acknowledgements

Thanks to [Finn Qiao](https://github.com/finnqiao) for providing QA and advice on this project as well as to Chloe Mawer, Fausto Inestroza, and Xiaofeng Zhu for their guidance and instruction in the **MSiA 423 - Analytics Value Chain** course.

## DataLinks

- "The Instacart Online Grocery Shopping Dataset 2017", Accessed from <https://www.instacart.com/datasets/grocery-shopping-2017> on 2019-04-10