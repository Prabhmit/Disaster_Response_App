
# importing libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # read in and merge files 
    messages = pd.read_csv('disaster_messages.csv')             
    categories = pd.read_csv('disaster_categories.csv')         
    
    # merge datasets
    df = messages.merge(categories,how='outer',on='id')
    
    return df 

def clean_data(df):
   
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,]

    # extract a list of new column names for categories.
    category_colnames = list(map(lambda x: x[:-2], row))

    # rename the columns of `categories`
    categories.columns = category_colnames

    # set each value to be the last character of the string
    for column in categories:
        categories[column] =  categories[column].apply(lambda x: x[-1])
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column]) 

    # drop the original, categories columns
    df.drop(labels=['categories','original'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)

    # Converting category columns to numeric
    df.iloc[:,3:] = df.iloc[:,3:].apply(pd.to_numeric)

    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # removing rows labelled as 2
    df.drop(df[df['related']==2].index, inplace=True)

    return df

def save_data(df, database_filepath):
    
    # load to database
    engine = create_engine('sqlite:///'+ database_filepath)
    df.to_sql('tidy_data', engine, if_exists='replace', index=False) 


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