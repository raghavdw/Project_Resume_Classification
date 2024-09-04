import os
from sklearn.linear_model import LogisticRegression
import joblib
from resume_model.config.core import Config

config = Config()

def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def save_model(model, vectorizer):
    # Ensure the trained_models directory exists
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    
    joblib.dump(model, config.model_path)
    joblib.dump(vectorizer, config.vectorizer_path)
