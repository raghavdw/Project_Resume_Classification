from resume_model.predict import predict_resume

def test_predict_resume():
    result = predict_resume("Sample resume text")
    assert result is not None
