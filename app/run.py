import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask, render_template, request

from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

# Initialize the Flask app
app = Flask(__name__)

# Define the tokenize function
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# Load the trained model
model = joblib.load("../models/classifier.pkl")

# Route for the main index page
@app.route('/')
@app.route('/index')
def index():
    # Create plotly visualizations for message genres and categories
    ids, graphJSON = create_visuals(df)
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Route for handling user queries and displaying model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '')

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the go.html template
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

# Define a function to create plotly visualizations
def create_visuals(df):
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = df.drop(['id', 'message', 'original', 'genre'], axis=1).sum(axis=0)
    category_names = category_counts.index

    # Create visualizations
    genre_graph = {
        'data': [
            Bar(
                x=genre_names,
                y=genre_counts,
                marker_color='skyblue'
            )
        ],
        'layout': {
            'title': 'Distribution of Message Genres',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Genre"
            }
        }
    }

    category_graph = {
        'data': [
            Bar(
                x=category_names,
                y=category_counts,
                marker_color='lightgreen'
            )
        ],
        'layout': {
            'title': 'Distribution of Message Categories',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Category",
                'tickangle': -45
            }
        }
    }

    # Create a list of graphs
    graphs = [genre_graph, category_graph]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return ids, graphJSON

# Run the app if this script is executed directly
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
