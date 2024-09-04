from resume_model.processing.features import extract_features

def test_extract_features(sample_data):
    features, _ = extract_features(sample_data)
    assert features.shape[1] == 1
