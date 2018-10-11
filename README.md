crypto-predict
==============================
[![Build Status](https://travis-ci.com/daniel-cortez-stevenson/crypto-predict.svg?branch=master)](https://travis-ci.com/daniel-cortez-stevenson/crypto-predict)
![Docker Build](https://img.shields.io/docker/automated/danielstevenson/crypto-predict.svg)
![Docker Build Status](https://img.shields.io/docker/build/danielstevenson/crypto-predict.svg)

This project is a easily reproducible python/flask/docker project to
create an API that predicts the price of BTC and ETH using OHLCV data.

Current implementation computes a continuous wavelet transformation
for the % change in price signal for the previous 72 hours. This data
is fed into a convolutional layer, which feeds into a dual layer LSTM
(network graph coming soon!).

***Note: Anaconda is recommended to manage the project environment. Environment creation without Anaconda is untested***

Make Commands
========

The Makefile contains the central entry points for common tasks related to this project.

Run the following to create the project python environment and get the training data from Cryptocompare API.

*from the top project directory*
```bash
make create_environment
source activate crypto-predict
make requirements
make data
```

Docker Usage
============
### Local
<i> from top directory </i>
```docker
docker build -f ./docker/Dockerfile -t crypto_predict_api .
docker run -p 5000:5000 crypto_predict_api
```
Now find your prediction at localhost:5000/predict?coin={ETH or BTC}

Future Directions
=================
- More coins!
- Visualize price predictions in a web app based on the developed prediction api
- Use unstructured text data to calculate sentiment or other metrics to use in prediction
- Use transfer learning from financial markets and other crypto coins to enhance the RNN model

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data`
    ├── README.md
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py
    ├── src
    │   │
    │   ├── CryptoPredict   <- main classes that utilize functions in other src/ directories.
    │   │
    │   ├── data           <- Functions to download or generate data
    │   │  
    │   │
    │   ├── features       <- Functions to turn raw data into features for modeling
    │   │  
    │   │
    │   ├── models         <- Functions to train models and then use trained models to make
    │   │                       predictions
    │   │  
    │   ├── visualization  <- Functions to create exploratory and results oriented visualizations
    │   │   
    │   │
    │   ├── tests   <- unit_test.py and unit_main.py
    │   │
    │   └── app.py <- the Flask code used to serve the prediction API
    │ 
    ├── docker  <- stores Dockerfiles for various deployments.
    │
    ├── .dockerignore   <- for all Dockerfiles. Ignores sensitive and large files.
    │
    ├── Dockerfile.dockerhub    <- Dockerfile used for hub.docker.com automated build
    │
    ├── .travis.yml     <- Automated continuous integration config
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
