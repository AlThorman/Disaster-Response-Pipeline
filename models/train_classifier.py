import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, hamming_loss

import pickle

def load_data(database_filepath):
    """
        Loads data from SQLite database.

        Parameters:
        database_filepath (str): Filepath to the database

        Returns:
        X (pd.Series): Features (messages)
        Y (pd.DataFrame): Target variables
        category_names (list): List of target category names
        """
    try:
        # Create a database engine
        engine = create_engine(f'sqlite:///{database_filepath}')

        # Load data from the 'disaster_messages' table in the database
        df = pd.read_sql_table("messages", con=engine)

        # Extract features (X) and target (Y)
        X = df['message']
        Y = df.iloc[:, 4:]  # Assuming the target variables start from column 4

        # Get category names
        category_names = Y.columns.tolist()

        return X, Y, category_names
    except Exception as e:
        print("An error occurred:", e)
        return None, None, None


def tokenize(text):
    """
        Tokenizes and lemmatizes text.

        Parameters:
        text (str): Text to be tokenized

        Returns:
        clean_tokens (list): Returns cleaned tokens
        """
    # Tokenize text
    tokens = word_tokenize(text)

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Get stopwords
    stop_words = set(stopwords.words('english'))

    # Iterate through each token
    clean_tokens = []
    for tok in tokens:
        # Lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        # Exclude stopwords and single characters
        if clean_tok not in stop_words and len(clean_tok) > 1:
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
        Builds a classifier and tunes the model using GridSearchCV.

        Returns:
        cv (GridSearchCV): Tuned classifier
        """
    # Create a pipeline with CountVectorizer, TF-IDF, and Random Forest Classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=1, random_state=42)))
    ])

    # Define hyperparameter grid for tuning
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 5]
    }

    # Create a GridSearchCV object with the pipeline and hyperparameters
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring='f1_micro', verbose=3, n_jobs=1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        Evaluates the performance of a model and returns a summary report.

        Parameters:
        model: Trained classifier
        X_test: Test dataset
        Y_test: Labels for the test data in X_test

        Returns:
        None (Prints a summary report)
        """
    y_pred = model.predict(X_test)

    print("Overall Metrics:\n")
    print("Accuracy:", accuracy_score(Y_test, y_pred))
    print("Hamming Loss:", hamming_loss(Y_test, y_pred))

    print("\nClassification Report for Each Column:\n")
    for column in Y_test.columns:
        print(f"Column: {column}")
        print(classification_report(Y_test[column], y_pred[:, Y_test.columns.get_loc(column)]))
        print("=" * 60)


def save_model(model, model_filepath):
    """
        Exports the final model as a pickle file.

        Parameters:
        model: Trained model
        model_filepath (str): Filepath to save the model

        Returns:
        None
        """
    try:
        # Serialize and save the model to the specified filepath
        with open(model_filepath, 'wb') as model_file:
            pickle.dump(model, model_file)
        print("Model saved successfully.")
    except Exception as e:
        print("An error occurred while saving the model:", e)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()