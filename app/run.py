import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import joblib


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Figure8Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    cat_names = df.columns[4:]
    cat_counts = df[cat_names].sum()
    
    #setup category groupings
    aid = ['medical_help', 'medical_products', 'death','search_and_rescue',
           'security','military','missing_people', 'refugees','water',
           'food','shelter','clothing','money','other_aid']
    infrastructure = ['infrastructure_related','transport','buildings',
                      'electricity','tools','hospitals','shops','aid_centers','other_infrastructure']
    weather = ['weather_related','floods','storm','fire','earthquake','cold','other_weather']
    general = ['related','request','offer','direct_report']

    #count messages with a at least one classification in the category group
    aid_df = df[df[aid] == 1]
    aid_df = aid_df.dropna(axis=0, how='all')
    aid_count = aid_df.sum(axis=1).count()

    infrastructure_df = df[df[infrastructure] == 1]
    infrastructure_df = infrastructure_df.dropna(axis=0, how='all')
    inf_count = infrastructure_df.sum(axis=1).count()

    weather_df = df[df[weather] == 1]
    weather_df = weather_df.dropna(axis=0, how='all')
    weather_count = weather_df.sum(axis=1).count()

    general_df = df[df[general] == 1]
    general_df = general_df.dropna(axis=0, how='all')
    general_count = general_df.sum(axis=1).count()
    
    cat_group_names = ['Aid Related', 'Infrastructure Related', 'Weather Related', 'General']
    cat_group_counts = [aid_count, inf_count, weather_count, general_count]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
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
        },
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Category Classifications',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cat_group_names,
                    y=cat_group_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Classification of messages into groups of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category Grouping"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()