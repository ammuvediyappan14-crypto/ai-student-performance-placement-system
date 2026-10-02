from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

PERFORMANCE_MODEL = MODEL_DIR / "performance_model.pkl"
PLACEMENT_MODEL = MODEL_DIR / "placement_model.pkl"


def create_dataset(number_of_students=1200, seed=42):
    rng = np.random.default_rng(seed)

    attendance = rng.uniform(45, 100, number_of_students)
    cgpa = rng.uniform(4.5, 10, number_of_students)
    internal_marks = rng.uniform(40, 100, number_of_students)
    projects = rng.integers(0, 6, number_of_students)
    skills_score = rng.uniform(35, 100, number_of_students)
    aptitude_score = rng.uniform(30, 100, number_of_students)
    communication_score = rng.uniform(35, 100, number_of_students)

    performance_score = (
        attendance * 0.18
        + cgpa * 8 * 0.25
        + internal_marks * 0.15
        + projects * 4 * 0.10
        + skills_score * 0.12
        + aptitude_score * 0.10
        + communication_score * 0.10
    )

    performance_label = np.where(
        performance_score >= 75,
        "Excellent",
        np.where(
            performance_score >= 60,
            "Good",
            np.where(
                performance_score >= 48,
                "Average",
                "Needs Improvement"
            )
        )
    )

    placement_score = (
        attendance * 0.10
        + cgpa * 10 * 0.20
        + internal_marks * 0.10
        + projects * 20 * 0.10
        + skills_score * 0.20
        + aptitude_score * 0.15
        + communication_score * 0.15
    )

    placement_label = (placement_score >= 68).astype(int)

    features = np.column_stack(
        [
            attendance,
            cgpa,
            internal_marks,
            projects,
            skills_score,
            aptitude_score,
            communication_score,
        ]
    )

    return features, performance_label, placement_label


def train_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, performance_y, placement_y = create_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        performance_y,
        test_size=0.20,
        random_state=42,
        stratify=performance_y,
    )

    performance_model = RandomForestClassifier(
        n_estimators=180,
        random_state=42,
        class_weight="balanced",
    )

    performance_model.fit(X_train, y_train)

    performance_prediction = performance_model.predict(X_test)

    performance_accuracy = accuracy_score(
        y_test,
        performance_prediction
    )

    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
        X,
        placement_y,
        test_size=0.20,
        random_state=42,
        stratify=placement_y,
    )

    placement_model = RandomForestClassifier(
        n_estimators=180,
        random_state=42,
        class_weight="balanced",
    )

    placement_model.fit(X_train_p, y_train_p)

    placement_prediction = placement_model.predict(X_test_p)

    placement_accuracy = accuracy_score(
        y_test_p,
        placement_prediction
    )

    with open(PERFORMANCE_MODEL, "wb") as file:
        pickle.dump(performance_model, file)

    with open(PLACEMENT_MODEL, "wb") as file:
        pickle.dump(placement_model, file)

    print(
        f"Performance model accuracy: "
        f"{performance_accuracy * 100:.2f}%"
    )

    print(
        f"Placement model accuracy: "
        f"{placement_accuracy * 100:.2f}%"
    )


if __name__ == "__main__":
    train_models()
