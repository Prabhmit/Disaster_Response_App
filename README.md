# Udacity Disaster Response Project

# Table of Contents

1. Installations and package requirement
2. Project Motivation
3. Project Components and File Descriptions
4. Instructions
5. Licensing, Authors and Acknowledgments

# Installations and package requirement

The code is in Python 3.7.6. The ETL Pipeline Preparation and ML Pipeline Preparation are Jupyter notebooks. The process_data.py, train_classifier.py and run.py are python scripts.The Python libraries applied are sys, nltk, sqlite3, numpy, pandas, pickle, sqlachemy and sklearn. 

# Project Motivation

The primary motivations are to analyze disaster data from Figure Eight to build a machine learning model for an API that classifies disaster messages for appropriate disaster relief agency. The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays data visualizations. 

# Project Components and File Descriptions

The datasets are messages.csv and categories.csv. The ETL Pipeline Preparation and ML Pipeline Preparation are Jupyter notebooks used to construct data cleaning and machine learning pipelines.

The project has three components: 

## 1. ETL Pipeline

The datasets were loaded, merged, cleaned and stored in a SQLite database. The code for this component is included in ETL Pipeline Preparation. 

## 2. ML Pipeline

The data was loaded from SQLite database. The datset was then split in training and test sets. A text processing and machine learning pipeline was build to train and tune the model using GridSearchCV. The model output was evaluated and the final model was exported as a pickle file. 

The text was cleand and tokenized and transformed to tfidf score. Sklearn's Random Forest Multi Output Classifier was used to build a model to classify disaster messages on 36 categories. The code for this component is included in ML Pipeline Preparation.

## 3. Flask Web App

The project includes a Flask Web App with Plotly data visualizations. The process_data.py script in data directory, includes modularised code to clean and store data in database named DisasterResponse. The datasets have been re-named disaster_messages.csv and disaster_categories.csv. The train_classifier.py script in models directory, includes modularised code to run the Machine Learning pipeline that trains classifier model and saves it as a pickle file. The run.py script in app directory contains relevant code to launch the web app and data visualizations.

# 4. Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

# 5. Licensing, Authors and Acknowledgments

Introduction to Data Science, Data Scientist Nanodegree Program, Udacity, https://www.udacity.com/course/data-scientist-nanodegree--nd025

Figure Eight, Appen, https://appen.com/

scikit-learn - sk.learn.ensemble.RandomForestClassifier, https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
