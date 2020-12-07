import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


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
df = pd.read_sql_table('tidy_data', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    # graph 1 - distribution of message genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # graph 2 - message category ratio
    categories = df[df.columns[3:]]   # subsetting category columns
    catg_percent = (categories.mean().sort_values(ascending=False)/categories.mean().sum())*100  # Normalized to be between 0 and 100
    catg_names = list(catg_percent.keys()) # category names
    
    # graph 3 -distribution of top four categories
    related = df[df['related']==1].groupby('genre').count()['related']  # category count by genre 
    aid_related = df[df['aid_related']==1].groupby('genre').count()['aid_related']
    weather_related = df[df['weather_related']==1].groupby('genre').count()['weather_related']
    direct_report = df[df['direct_report']==1].groupby('genre').count()['direct_report']
    

    # create visuals
    graphs = [
        { # graph 1
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Genre"}
            }
        },
   
    { 
        'data' : [# graph 2
                    Bar(
                        y=catg_percent,
                        x=catg_names
                    )
          ],
        
         'layout' : {
                    'title' : 'Message Category Ratios',
                    'xaxis' : {'title' : '','tickangle': 25},
                    'yaxis' : {'title' : 'Percent'}
                    }
        },
       
        {
            'data': [#graph - 3
                Bar(
                    x=genre_names,
                    y=related,
                    name= 'related'
                    ),
                
                Bar(
                    x=genre_names,
                    y=aid_related,
                    name='aid_related'
                ),
                
                Bar(
                    x=genre_names,
                    y=weather_related,
                    name='weather_related'
                 ),
               
                Bar(
                    x=genre_names,
                    y=direct_report,
                    name='weather_related'
                )
            ],
            
            'layout': {
                'title': 'Genre distribution for top four categories',
                'yaxis': {'title': "Count"},
                'xaxis': {'title': "Categories"},
                'barmode' : 'group'
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