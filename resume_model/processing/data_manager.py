import pandas as pd
from resume_model.config.core import Config

config = Config()

def load_data():
    data = pd.read_csv(config.data_path)
    return data

def preprocess_data(data):
    # Implement preprocessing steps here (e.g., encoding, cleaning)
    return data
