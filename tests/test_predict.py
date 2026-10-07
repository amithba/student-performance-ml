import os
import joblib
import pandas as pd


def test_model_file_exists():
    assert os.path.exists("models/model.pkl")


def test_model_prediction():
    model = joblib.load("models/model.pkl")

    student = pd.DataFrame(
        [[6, 85, 70, 75]],
        columns=[
            "study_hours",
            "attendance",
            "previous_marks",
            "assignment_score"
        ]
    )

    prediction = model.predict(student)[0]

    assert prediction in [0, 1]