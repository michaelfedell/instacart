# Data

The data contained in this folder is made up of raw, external data, hand-crafted auxiliary data, and code-generated, feature-rich data for the modeling pipeline.

## Auxiliary

Additional data created or added during development

- **cats.yml**: This file maps grocery aisles present in the dataset to high-level macro categories. This was done by manually inspecting all unique aisles and considering them as one or more of a discrete set of categories.

## External

Source data downloaded directly from Instacart

- **aisles.csv**: names of aisles in store
- **data_description.md**: describes each column in dataset
- **departments.csv**: names of departments in store
- **order_products__prior.csv**: Transactional information about each product in a given order
- **order_products__train.csv**: Transactional information about each product in a given order (only most recent order of each user)
- **orders.csv**: Transactional information about each order placed
- **products.csv**: All possible products which may be ordered
- **raw_data.tar.gz**: Raw data downloaded from Instacart which contains all the above files

## Features

Processed data which has been produced from the raw data described above. This data will be passed forward in the model pipeline.

- **baskets.csv**: The augmented and processed orders data from orders.csv - includes label assignments from clustering
- **order_types.csv**: Summary information describing each of the order types as defined by the clustering labels produced in feature generation. An order_type cluster is characterized mainly by the attributes of its centroid
- **shoppers.csv**: Profiles built for each of the users in the dataset. Includes behavioral and transactional summary statistics
