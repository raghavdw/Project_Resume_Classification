from sklearn.metrics import classification_report

def validate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    return report
