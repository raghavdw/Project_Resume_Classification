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
    try:
        model = joblib.load(config.model_path)
        vectorizer = joblib.load(config.vectorizer_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

@app.post("/predict")
def predict(resume: Resume):
    try:
        features = vectorizer.transform([resume.resume_text])
        prediction = model.predict(features)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
