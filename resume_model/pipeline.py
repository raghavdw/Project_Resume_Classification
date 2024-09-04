from resume_model.processing.data_manager import load_data, preprocess_data
from resume_model.processing.features import extract_features
from resume_model.trained_models.model import train_model, save_model
from resume_model.config.core import Config
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def run_pipeline():
    # Load configuration
    config = Config()
    
    # Step 1: Load and preprocess data
    data = load_data()  # Load your data (you need to implement or check this)
    data = preprocess_data(data)  # Preprocess data (you need to implement or check this)
    
    # Step 2: Feature extraction using TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=config.tfidf_max_features)
    features = vectorizer.fit_transform(data['Resume'])  # Fit and transform the data

    # Step 3: Train the model
    model = train_model(features, data['Category'])  # Train your model (you need to implement or check this)

    # Step 4: Save the trained model and the vectorizer
    save_model(model, vectorizer)  # Ensure this function saves both the model and vectorizer
    
    # Step 5: Save vectorizer separately if needed
    joblib.dump(vectorizer, config.vectorizer_path)

if __name__ == "__main__":
    run_pipeline()
