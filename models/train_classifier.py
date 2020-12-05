# importing necessary libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import sqlite3
import re 
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier


def load_data(database_filepath):
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql("SELECT * from tidy_data", conn)
    conn.close()
    
    # define features and label arrays
    X = df.iloc[:,1]      #.values
    y = df.iloc[:,3:]     #.values
    category_names = list(df.iloc[:,3:].columns)

    return X, y, category_names

def tokenize(text):
    # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate stop words
    stop_words = stopwords.words("english")
    
    # remove stop words
    tokens = [t for t in tokens if t not in stop_words]
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    
    # Build a pipeline, Note: classes are imbalanced
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),('tfidf',TfidfTransformer()), 
                     ('clf',MultiOutputClassifier(RandomForestClassifier(class_weight='balanced',n_jobs=-1)))])  
    
    # Using grid search to find better parameters   # Make changes to this
    parameters =  {#'vect__max_df': (0.2,0.3,0.5),
               #'vect__ngram_range': ((1,1),(1,2),(2,2)),
               #'tfidf__use_idf': (True, False),
               #'clf__estimator__max_depth': [3,4,5],          # Keep this one
               'clf__estimator__min_samples_split': [3,5]}     # CHANGE THIS
    
    # Create grid search object
    model = GridSearchCV(pipeline, param_grid=parameters) #,scoring='f1'
        
    return model


def evaluate_model(model, X_test, y_test, category_names):
    
    # Predict on test data
    y_pred = model.predict(X_test)
   
    #for i in range(0,36):
    #    print('category:',category_names[i])
    #    print(classification_report(y_test[:,i],y_pred[:,i]))


def save_model(model, model_filepath):
    pickled_filename = 'trained_model.pkl'
    pickle.dump(model, open(model_filepath + pickled_filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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