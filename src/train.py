import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Load dataset
data = pd.read_csv("data/student_data.csv")

# Input features
X = data[
    [
        "study_hours",
        "attendance",
        "previous_marks",
        "assignment_score"
    ]
]

# Target
y = data["result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# MLflow experiment
mlflow.set_experiment("Student Performance Prediction")

with mlflow.start_run():

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    # Log parameters
    mlflow.log_param("algorithm", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("test_size", 0.2)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Create models folder
    os.makedirs("models", exist_ok=True)

    # Save model
    joblib.dump(
        model,
        "models/model.pkl"
    )

    # Log model to MLflow
    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

print("Model training completed successfully.")