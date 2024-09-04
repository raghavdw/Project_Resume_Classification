from sklearn.feature_extraction.text import TfidfVectorizer
from resume_model.config.core import Config

config = Config()

def extract_features(data):
    vectorizer = TfidfVectorizer(max_features=config.tfidf_max_features)
    features = vectorizer.fit_transform(data['Resume'])
    return features, vectorizer
