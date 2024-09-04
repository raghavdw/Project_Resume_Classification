import joblib
from resume_model.config.core import Config

config = Config()

def predict_resume(resume_text):
    model = joblib.load(config.model_path)
    vectorizer = joblib.load(config.vectorizer_path)
    
    features = vectorizer.transform([resume_text])
    prediction = model.predict(features)
    return prediction
