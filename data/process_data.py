import pandas as pd
import numpy as np
import re
import sqlite3
from sqlalchemy import create_engine
import sys

'''
Purpose:
    1. Load messages.csv and categories.csv
    2. Clean / process the data into a SQL table resembling the format: 'ID',
        'Message', 'Original;, ' Genre', 'Cat1', ..., 'CatN'
    3. Store the table into a SQL DB with the provided filename
Usage:
    Call with: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

Inputs:
    messages_filepath: (String) Path / Location of the CSV file containing the message data
    categories_filepath: (String) Path / Location of the CSV file containing the category classification data
    database_filepath: (String) Path / Location for the SQL DB containing the disaster response table
    
Outputs:
    None
    
Results:
    Database file created at the specified location using the specified input data CSV files
'''

def load_data(messages_filepath, categories_filepath):
    '''
    Purpose: Load the messages and categories CSV files into DataFrames
    Inputs:
        messages_filepath: (String) Location of the messages CSV file
        categories_filepath: (String) Location of the categories CSV file
    Outputs:
        df_combined: (DataFrame) df containing the messages data merged with the categories data
    '''
    
    # read the CSV files
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df_combined = df_messages.merge(df_categories, how='outer', on='id')
    
    return df_combined

def clean_data(df):
    '''
    Purpose: Process the data to remove or correct problematic entries such as NANs, unexpected characters, etc.
    Inputs:
        df: (DataFrame) DF containing the merged data set
    Outputs:
        df_clean: (DataFrame) DF containing the expanded and cleaned merged dataset with individual category columns
    '''
    
    # create a dataframe of the 36 individual category columns
    split_categories = df['categories'].str.split(';')
    
    # select the first row of the categories dataframe
    row = pd.Series(split_categories[0])

    # use this row to extract a list of new column names for categories.
    # using lambda apply, remove the end of the category and retain just the name
    category_colnames = row.apply(lambda x: x[0:-2])
    
    # rename the columns of `categories`
    categories = pd.DataFrame(columns=category_colnames)
    
    #for row in the dataset:
    # set each value to be the last character of the string
    for i in range (0, len(split_categories)):
        row = pd.Series(split_categories[i],index = category_colnames).apply(lambda x: x[-1])
        categories = categories.append(row, ignore_index = True)
    
    #for column in categories:
    for column in categories:
    	# convert column from string to numeric
    	categories[column] = pd.to_numeric(categories[column])
    categories = categories.mask(categories > 1,1)

    # drop the original categories column from `df`
    df = df.drop(['categories'],axis=1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # remove any extra whitespace
    df['message'].replace(' +', ' ', inplace=True,regex=True)
    
    # identify duplicates and drop all but the first
    df = df.drop_duplicates(subset=['message'],keep='first')
    
    return df

def save_data(df, database_filename):
    '''
    Purpose:
        To generate a SQL database with the filename specified and upload the dataframe
    Inputs:
        df: (DataFrame) DF containing the cleaned messages/categories data
        database_filename: (String) Filename (not path) for the DB
    Outputs:
        None
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Figure8Messages', engine, if_exists='replace', index=False)  

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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()