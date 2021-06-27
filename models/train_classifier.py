import sys
# import libraries
import sqlite3
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB

from joblib import dump

'''
Purpose:
    1. Load SQL database at the specified path
    2. Build, train, and predict on a model using the DB
    3. Evaluate the model
    4. Save the model as a pkl file in the specified location
Usage:
    Call with: 'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'
Inputs:
    database_filepath: (String) Path / Location for the SQL DB containing the disaster response table  
    model_filepath: (String) Path / Location for the model pkl file to be saved
Outputs:
    None    
Results:
    Model built, trained, predicted on, and saved
'''

def load_data(database_filepath):
    '''
    Purpose:
        Load the Database at the specified path and sort into X and Y datasets
    Inputs:
        database_filepath: (String) Path to the database
    Outputs:
        X: (DataFrame) DF containing the message text data
        Y: (DataFrame) DF containing the various category classifications
        Y.columns: (List of Strings) List of the categories present in the Y data
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Figure8Messages',engine.connect())
    
    # split into X (message text) and Y (categories)
    X = df[['message']]
    X = X.message.tolist()
    Y = df.drop(axis=1, columns=['id', 'message', 'original','genre'])
    
    # Drop any categories that have no entry (ie, in the provided DB, "child_alone" was associated with none of the messages
    for col in Y.columns:
        if Y[col].max() == 0:
            Y = Y.drop(axis=1, columns=[col])
    
    return X, Y, Y.columns

def tokenize(text):
    '''
    Purpose:
        To accept text data, normalize to lowercase, remove stopwords, stem / lemmatize and split into tokens
    Input:
        text: (String) Text data of a message
    Outputs:
        clean_tokens: (List of strings) List of single word strings (tokens) based on the input, text
    '''
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #split into tokens
    tokens = word_tokenize(text)
    
    #initialize stop words, lemmatizer and stemmers
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    stemmer = PorterStemmer()
    
    #reduce tokens into just the interesting words by removing
    # stop words and lemmatizing the words before then stemming them
    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = stemmer.stem(lemmatizer.lemmatize(tok))
            clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    '''
    Purpose:
        Create a semi-optimized pipeline object that can be fit based off of the X and Y data
    Inputs:
        None
    Outputs:
        model: (Pipeline) model based on previously gridsearched parameters, CountVectorizer, TfidTransformer,
                and MultiOutputClassifier version of RandomForestClassifier / MultinomialNB
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.75)),
        ('tfidf', TfidfTransformer(sublinear_tf=False)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=500, min_samples_leaf=1)))
    ])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Purpose:
        To predict values based on the test set and print out model performance / prediction
    Inputs:
        model: (Pipeline) Pipeline that has been fit based on the X and Y Training sets
        X_test: (DataFrame) DF containing the test input message data
        Y_test: (DataFrame) DF containing the test output category data
        category_names: (List of Strings) List of the column names in Y
    Outputs:
        None
    '''
    
    #Predict
    Y_pred = model.predict(X_test)
    
    #Generate the confusion matrices and classification reports
    for idx, col in enumerate(category_names):
        print(col)
        confusion_mat = confusion_matrix(Y_test[col], Y_pred[:,idx], labels=np.unique(Y_pred[:,idx]))
        print('----------------------------------')
        print(classification_report(Y_test[col], Y_pred[:,idx]))
        print("Confusion Matrix:\n\n", confusion_mat)
        print("Accuracy: {}".format((Y_test[col] == Y_pred[:,idx]).mean()))
        print("\n")

def save_model(model, model_filepath):
    '''
    Purpose:
        Save the model to the specified path
    Inputs:
        model: (Pipeline) Pipeline based on the previosuly trained data set
        model_filepath: (String) Path to save the model to
    Outputs:
        None
    '''
    dump(model, model_filepath)


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