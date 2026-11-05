
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_pipeline(data_path, model_path):

    # Load dataset
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    # Clean target column
    df["classification"] = (
        df["classification"].astype(str).str.strip().str.lower()
    )

    df["classification"] = df["classification"].replace({
        "ckd": 1,
        "notckd": 0
    })

    df = df[df["classification"].isin([0, 1])]

    # Separate features and target
    X = df.drop(
        columns=["classification", "id"], errors="ignore"
    )
    y = df["classification"].astype(int)

    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(exclude=["number"]).columns

    # Numerical preprocessing
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # Create end-to-end pipeline
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Pipeline executed successfully!")
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Model Accuracy:", accuracy)

    # Save complete pipeline
    joblib.dump(pipeline, model_path)
    print("Pipeline saved successfully!")

    # Load saved pipeline
    loaded_pipeline = joblib.load(model_path)

    # Perform prediction
    sample = X_test.iloc[[0]]
    prediction = loaded_pipeline.predict(sample)

    print("Sample Prediction:", prediction[0])

    return pipeline

# Execute pipeline
if __name__ == "__main__":
    run_pipeline("kidney_cleaned.csv", "kidney_pipeline.pkl")
