from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from resume_model.config.core import Config

app = FastAPI()

config = Config()

# Pydantic model to define the request body structure
class Resume(BaseModel):
    resume_text: str

@app.on_event("startup")
def load_model():
    global model, vectorizer
    model = joblib.load(config.model_path)  # Load the pre-trained model
    vectorizer = joblib.load(config.vectorizer_path)  # Load the vectorizer

@app.post("/predict")
def predict(resume: Resume):
    try:
        features = vectorizer.transform([resume.resume_text])  # Transform the input resume text
        prediction = model.predict(features)  # Make a prediction
        return {"prediction": prediction[0]}  # Return the prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
    
@app.get("/")
def read_root():
    return {"message": "Welcome to the Resume Classification API"}
