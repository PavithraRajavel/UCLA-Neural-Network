import pandas as pd
import logging


def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        logging.info(f'Data loaded successfully from {filepath}')
        return data
    except Exception as e:
        logging.error(f'Error loading data from {filepath}: {e}')
        raise


def preprocess_data(data):
    try:
        data['Admit_Chance'] = (data['Admit_Chance'] >= 0.8).astype(int)
        data = data.drop(['Serial_No'], axis=1)
        logging.info('Data preprocessed successfully')
        return data
    except Exception as e:
        logging.error(f'Error preprocessing data: {e}')
        raise
