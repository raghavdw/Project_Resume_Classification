import yaml
import os

class Config:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), '../config.yml')
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    @property
    def data_path(self):
        return self.config['data_path']

    @property
    def model_path(self):
        return self.config['model_path']

    @property
    def vectorizer_path(self):
        return self.config['vectorizer_path']

    @property
    def model_type(self):
        return self.config['model_type']

    @property
    def tfidf_max_features(self):
        return self.config['tfidf_max_features']

    @property
    def logging_level(self):
        return self.config['logging_level']

    @property
    def train_test_split_ratio(self):
        return self.config['train_test_split_ratio']

    @property
    def random_seed(self):
        return self.config['random_seed']

# Example usage
config = Config()
print(config.data_path)
