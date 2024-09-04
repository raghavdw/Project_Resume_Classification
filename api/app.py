from flask import Flask, request, jsonify
import joblib
from resume_model.config.core import Config

app = Flask(__name__)

config = Config()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    resume_text = data.get('resume_text', '')

    model = joblib.load(config.model_path)
    vectorizer = joblib.load(config.vectorizer_path)

    features = vectorizer.transform([resume_text])
    prediction = model.predict(features)

    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
