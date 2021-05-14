import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import string
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords'])
nltk.download('averaged_perceptron_tagger')
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    '''
    Function that loads data from SQL and creates input and targets. 
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df', engine)
    # drop target feature 'child_alone' since it only contains one class
    df = df.drop('child_alone', axis=1)
    #convert all categories to binary
    df['related'] = df['related'].astype('str').str.replace('2', '1')
    df['related'] = df['related'].astype('int')
    # split into targets and feature
    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    categories = list(Y.columns.values)
    return X, Y, categories

def tokenize(text):
    '''
    Function that tokenizes the text input by words and applies a lemmatizer.
    Inputs: text 
    Outputs: clean_words
    '''
    # tokenize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = nltk.corpus.stopwords.words("english")
    clean_words= [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word not in stop_words]
    return clean_words

def build_model():
    '''
    Function that builds the pipeline. 
    Input:  message column as input 
    Output: pipeline with customer tokenizer and multi output classifier
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=1)))
    ])

    parameters = {'clf__estimator__criterion': ['gini', 'entropy'],
                  'clf__estimator__n_estimators': [50, 100],
                  'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.5, 1.0)
                  }
    #run for shorter times
    parameters2 = {
        'vect__max_df': (0.5, 1.0),
        'clf__estimator__n_estimators': [10, 100],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters2, n_jobs=-1,verbose=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function that computes f1, recall and precision scores for test set predictions
    Inputs: 
    model: model from build_model() function specifications
    X_test: messages
    Y_test: categories
    category_names: list of category names 
    Outputs: None
    '''
    # predict on test data
    y_pred = model.predict(X_test)
    print(model.best_params_, model.best_score_)
    for i, col in enumerate(category_names):
        print(i, col)
        print(classification_report(Y_test.to_numpy()[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    ''' 
    Function that saves model as pickle object 
    Inputs: 
    model: model from build_model() function specifications
    model_filepath: model file path, e.g. "models/model.p.gz"
    Output: None
    '''
    pickle.dump(model,open(model_filepath,'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...n    DATABASE {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42, shuffle=True)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...n    MODEL {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. nnExample python '
              'train_classifier.py ..dataDisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
#python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
