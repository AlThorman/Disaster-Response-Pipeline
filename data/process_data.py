import sys
import pandas as pd
from sqlalchemy import *

def load_data(messages_filepath, categories_filepath):
    """
        Loads and merges 2 datasets.

        Parameters:
        messages_filepath: messages csv file
        categories_filepath: categories csv file

        Returns:
        df: dataframe containing messages_filepath and categories_filepath merged

        """


    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id', how='outer')
    return df


def clean_data(df):
    """
        Cleans the dataframe.

        Parameters:
        df: DataFrame

        Returns:
        df: Cleaned DataFrame

        """

    # Split categories into separate columns
    categories = df['categories'].str.split(";", expand=True)

    # Create more descriptive column names by removing subcategories
    category_colnames = [col.split("-")[0] for col in categories.iloc[0]]
    categories.columns = category_colnames

    # Convert category values to binary (0 or 1)
    categories = categories.applymap(lambda x: 1 if x[-1] == '1' else 0)

    # Remove the original 'categories' column
    df = df.drop(columns=['categories'])

    # Concatenate the modified categories DataFrame with the original DataFrame
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates based on all columns
    df = df.drop_duplicates(keep='first')

    return df


def save_data(df, database_filename):
    '''
        Saves the cleaned DataFrame to an SQLite database table named 'messages'.

        Parameters:
            df (pandas.DataFrame): The cleaned DataFrame to be saved.
            database_filename (str): The name of the SQLite database file.
        '''

    engine = create_engine('sqlite:///' + database_filename)

    try:
        # Save the DataFrame to the 'messages' table in the database
        df.to_sql('messages', engine, index=False, if_exists='replace')
        print("Data saved to the 'messages' table.")
    except Exception as e:
        print("An error occurred:", e)

    df.to_sql('messages', engine, index=False)


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