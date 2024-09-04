from resume_model.predict import predict_resume

def test_predict_resume():
    """ sample prediction function"""
    
    result = predict_resume("law and order")
    assert result is not None
