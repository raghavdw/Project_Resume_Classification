from resume_model.predict import predict_resume

def test_predict_resume():
    result = predict_resume("law and order")
    assert result is not None
