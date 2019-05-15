Built by [Michael Fedell](https://github.com/michaelfedell), QA'd by [Finn Qiao](https://github.com/finnqiao)

# InstaCart

InstaCart has released rich historical data on the grocery shopping habits of their customers. We will use this data to create profiles with predictive power which will help users discover products and stores manage planning and logistical challenges.

Check out the [Project Charter](CHARTER.md) for some background on this project's inception.

Or, to see the planned work, check out the TODO: [issues]() or [ZenHub Board]()

# Charter

This project was started as a way to enrich the grocery shopping experience for both consumers and suppliers. The project also serves as a testing ground for various tools and technologies such as cloud computing, machine-learning driven recommendation systems, application deployment and maintenance, as well as basic web development in the Flask-python ecosystem.

## Vision

To enrich the grocery shopping experience by bringing added convenience and guidance to shoppers, confidence and insight to suppliers, and a delightful interface to all.

## Mission

Learn from experienced grocery shoppers and provide the resulting guidance in a sleek application that makes it easy for users to identify similar shoppers, benchmark their own shopping habits and discover new ideas while simultaneously helping grocery stores manage expectations around revenue, store traffic, and supply needs.

## Success Criteria

TODO:

## Planned Work

TODO: [Click here]() for the live issue board

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
├── config                            <- Directory for yaml configuration files for model training, scoring, etc
│   ├── logging/                      <- Configuration files for python loggers
│
├── data                              <- Folder that contains data used or generated. Only the external/ and sample/ subdirectories are tracked by git. 
│   ├── archive/                      <- Place to put archive data is no longer used. Not synced with git.
│   ├── external/                     <- External data sources, will be synced with git
│   ├── sample/                       <- Sample data used for code development and testing, will be synced with git
│
├── docs                              <- A default Sphinx project; see sphinx-doc.org for details.
│
├── figures                           <- Generated graphics and figures to be used in reporting.
│
├── models                            <- Trained model objects (TMOs), model predictions, and/or model summaries
│   ├── archive                       <- No longer current models. This directory is included in the .gitignore and is not tracked by git
│
├── notebooks
│   ├── develop                       <- Current notebooks being used in development.
│   ├── deliver                       <- Notebooks shared with others.
│   ├── archive                       <- Develop notebooks no longer being used.
│   ├── template.ipynb                <- Template notebook for analysis with useful imports and helper functions.
│
├── src                               <- Source scripts for the project
│   ├── archive/                      <- No longer current scripts.
│   ├── helpers/                      <- Helper scripts used in main src files
│   ├── sql/                          <- SQL source code
│   ├── ingest_data.py                <- Script for ingesting data from different sources
│   ├── generate_features.py          <- Script for cleaning and transforming data and generating features used for use in training and scoring.
│   ├── train_model.py                <- Script for training machine learning model(s)
│   ├── score_model.py                <- Script for scoring new predictions using a trained model.
│   ├── postprocess.py                <- Script for postprocessing predictions and model results
│   ├── evaluate_model.py             <- Script for evaluating model performance
│
├── test                              <- Files necessary for running model tests (see documentation below)

├── run.py                            <- Simplifies the execution of one or more of the src scripts
├── app.py                            <- Flask wrapper for running the model
├── config.py                         <- Configuration file for Flask app
├── requirements.txt                  <- Python package dependencies
```

## Documentation

## Getting Started

In this phase of the project, all raw data exist in CSV's as downloaded from Instacart [as linked below](##DataLinks). In order to get up and running yourself, you will need to download these large files and move all `.csv` files into `./data/external/`

## Testing

## Acknowledgements

## DataLinks

- "The Instacart Online Grocery Shopping Dataset 2017", Accessed from <https://www.instacart.com/datasets/grocery-shopping-2017> on 2019-04-10