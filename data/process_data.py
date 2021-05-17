import sys
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function that load disaster messages and categories cvs files into one merged DataFrames
    messages_filepath, categories_filepath: filepath of the cvs files
    Output:
    df: merged DataFrames
    '''

    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    #Merge the messages and categories datasets using the common id
    df = pd.merge(messages, categories, on="id")
    return df

def clean_data(df):
    '''
    Function that cleans the data for model input.
    Input:
    df: DataFrame
    Output:
    df: Cleaned DataFrame
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    # grab the first row for the header
    row = categories[1:]
    # set the header row as the df header
    # use this row to extract a list of new column names for categories.
    category_colnames = row.applymap(lambda s: s[:-2]).iloc[0, :].tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames
    #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0).
    #Convert the string to a numeric value.
    categories = categories.applymap(lambda s: int(s[-1])).astype(int)
    #convert all categories to binary
    # Value 2 also means no to do question and replaced with 0
    categories.related.replace(2, 0, inplace=True)
    # drop the original categories column from `df`
    df=df.drop(columns=['categories'])
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(subset='message', inplace=True)
    # DropNAs
    df.dropna(subset=category_colnames, inplace=True)
    return df

def save_data(df, database_filename):
    '''
    Function that saves DataFrame to SQL.
    Inputs:
    df: DataFrame
    database_filename: SQL database filename
    Output:
    None
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('df', engine,index=False, if_exists='replace', )

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
