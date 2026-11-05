

import pandas as pd
import os
import joblib

# Test 1: Check dataset
def test_dataset():
    df = pd.read_csv("kidney_cleaned.csv")
    assert not df.empty
    assert "classification" in df.columns
    print("Dataset test passed!")

# Test 2: Check model file
def test_model():
    assert os.path.exists("kidney_pipeline.pkl")
    model = joblib.load("kidney_pipeline.pkl")
    assert model is not None
    print("Model loading test passed!")

# Test 3: Check prediction
def test_prediction():
    df = pd.read_csv("kidney_cleaned.csv")
    df.columns = df.columns.str.strip()

    X = df.drop(
        columns=["classification", "id"], errors="ignore"
    )

    model = joblib.load("kidney_pipeline.pkl")
    predictions = model.predict(X.head(5))

    assert len(predictions) == 5
    print("Prediction test passed!")

if __name__ == "__main__":
    test_dataset()
    test_model()
    test_prediction()
    print("All tests passed!")
