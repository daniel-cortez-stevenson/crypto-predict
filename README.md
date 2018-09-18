crypto-predict
==============================
==============================

This project is a easily reproducible python/docker project to predict the price of bitcoin at the next hours using OHLCV data for bitcoin from the last 144 hours.

Commands
========

The Makefile contains the central entry points for common tasks related to this project.

Run the following to create the project python environment and get the data from cryptocompare API.

With Anaconda installed ...
*from the top directory*
```bash
make create_environment
source activate crypto-predict
make requirements
make data
```
Without Anaconda installed ...
You can figure it out - you're an advanced python user!

Docker Usage
=========
<i> from top directory (with Dockerfile in it)</i>
```docker
docker build -t crypto_predict_api .
docker run -p 5000:5000 crypto_predict_api
```
Now find your prediction at localhost:5000/predict?coin=BTC

Future Directions
=================
- More coins!
- Visualize price predictions in a web app based on the developed prediction api
- Use unstructured text data to calculate sentiment or other metrics to use in prediction
- Use transfer learning from financial markets and other crypto coins to enhance the RNN model

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── make_dataset.py
    |   |   └── get_data.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   ├── train_model.py
    |   │   └── load_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py
    │   │
    │   └── app.py <- the Flask code used to serve the prediction API
    │ 
    └── Dockerfile
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
