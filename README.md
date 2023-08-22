# Disaster Response Pipeline

## Table of Contents
- [Instructions](#Instructions)
- [Project Motivation](#Project-Motivation)
- [File Description](#File-Description)
- [Results](#Results)
- [Licensing, Authors, Acknowledgements](#Licensing,-Authors,-Acknowledgements)

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Motivation
This project aims to create a model and web app that classify disaster messages. Emergency workers can quickly input a message and receive categorization results like "water," "shelter," and "food." This helps responders determine needed assistance promptly and allocate resources effectively during disasters.

## File Description
- app: This directory contains the web application's templates, where master.html is the main page and go.html displays the classification results. The run.py file is responsible for running the Flask app.

- data: Here, you'll find the raw data files disaster_categories.csv and disaster_messages.csv. The process_data.py script handles the data cleaning pipeline, and InsertDatabaseName.db is the database where the clean data is stored.

- models: In this directory, train_classifier.py contains the machine learning pipeline for training the model. The trained model is saved as classifier.pkl.

- README.md: The project's README file provides essential information about the project, its purpose, and how to run it.

## Results
![1](https://github.com/AlThorman/Disaster-Response-Pipeline/assets/33887084/6d6a313a-5b3b-4743-9bf3-1e68874fc433)

![2](https://github.com/AlThorman/Disaster-Response-Pipeline/assets/33887084/aa74063f-01a9-4d7e-9306-568709c8038a)

![3](https://github.com/AlThorman/Disaster-Response-Pipeline/assets/33887084/f392a76d-3777-4316-83b2-bee33851f994)

## Licensing, Authors, Acknowledgements
The data used for this project was originally sourced from Figure Eight and is publicly available for research and development.
