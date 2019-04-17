# Smart Cart

Smart Cart is a comprehensive grocery list manager that offers smart recommendations on things you may want to add to your list!

Check out the [Project Charter](CHARTER.md) for some background on this project's inception.

Or, to see the planned work, check out the [issues](https://github.com/michaelfedell/smart_cart/issues) or [ZenHub Board](https://github.com/michaelfedell/smart_cart#workspaces/smart-cart-5cae419280656854a0156607/board?repos=180641233)

# Charter

This project was started as a way to improve the grocery shopping experience, especially for young people who are still learning to cook. The project also serves as a testing ground for various tools and technologies such as cloud computing, machine-learning driven recommendation systems, application deployment and maintenance, as well as basic web development in the Flask-python ecosystem.

## Vision

To create a simple and fun experience out of grocery shopping which inspires creativity and exploration.

## Mission

Learn from experienced cooks and grocery shoppers and provide the resulting guidance in a sleek application that makes it easy for users to manage grocery lists, find new recipe ideas, and share their findings with friends.

## Success Criteria

- [] 30% utilization of all served recommendations
- [] 75% of users adopt recommendations
- [] ≥ 80% positive feedback on UX design
- [] manage grocery lists and gather feedback from at least 20 total users

## Planned Work

[Click here](https://github.com/michaelfedell/smart_cart/issues) for the live issue board

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
│   ├── archive/                      <- Place to put archive data is no longer usabled. Not synced with git.
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
├── src                               <- Source data for the project
│   ├── archive/                      <- No longer current scripts.
│   ├── helpers/                      <- Helper scripts used in main src files
│   ├── sql/                          <- SQL source code
│   ├── add_songs.py                  <- Script for creating a (temporary) MySQL database and adding songs to it
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

## Testing

## Acknowledgements

## DataLinks

- "The Instacart Online Grocery Shopping Dataset 2017", Accessed from <https://www.instacart.com/datasets/grocery-shopping-2017> on 2019-04-10