"""Make predictions using the trained student performance model."""

import joblib
import pandas as pd


# Load trained model
model = joblib.load("models/model.pkl")

# Example student data
student = pd.DataFrame(
    [[6, 85, 70, 75]],
    columns=[
        "study_hours",
        "attendance",
        "previous_marks",
        "assignment_score"
    ]
)

# Make prediction
prediction = model.predict(student)[0]

# Display result
if prediction == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")
