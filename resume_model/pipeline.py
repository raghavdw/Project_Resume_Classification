from resume_model.processing.data_manager import load_data, preprocess_data
from resume_model.processing.features import extract_features
from resume_model.trained_models.model import train_model, save_model
from resume_model.config.core import Config

def run_pipeline():
    config = Config()
    data = load_data()
    data = preprocess_data(data)
    features, vectorizer = extract_features(data)
    model = train_model(features, data['Category'])
    save_model(model, vectorizer)

if __name__ == "__main__":
    run_pipeline()
