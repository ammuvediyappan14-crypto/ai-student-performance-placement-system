import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

PERFORMANCE_MODEL = MODEL_DIR / "performance_model.pkl"

PLACEMENT_MODEL = MODEL_DIR / "placement_model.pkl"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "attendance",
    "cgpa",
    "internal_marks",
    "projects",
    "skills_score",
    "aptitude_score",
    "communication_score"
]


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # SAMPLE TRAINING DATA
    # --------------------------------------------------------

    X = np.array([
        [95, 9.2, 92, 5, 90, 88, 92],
        [90, 8.8, 86, 4, 85, 84, 88],
        [85, 8.2, 80, 4, 80, 78, 82],
        [80, 7.8, 75, 3, 75, 72, 78],
        [75, 7.2, 70, 3, 68, 65, 70],
        [70, 6.8, 65, 2, 62, 60, 65],
        [65, 6.2, 60, 2, 55, 52, 58],
        [60, 5.8, 55, 1, 50, 48, 52],
        [55, 5.2, 50, 1, 45, 42, 48],
        [50, 4.8, 45, 0, 40, 38, 42],
        [92, 9.0, 90, 5, 88, 90, 91],
        [88, 8.5, 84, 4, 82, 80, 85],
        [82, 7.9, 78, 3, 76, 74, 79],
        [78, 7.5, 73, 3, 70, 68, 73],
        [72, 7.0, 68, 2, 64, 62, 68],
        [68, 6.5, 63, 2, 58, 56, 62],
        [62, 6.0, 58, 1, 52, 50, 55],
        [58, 5.5, 52, 1, 48, 45, 50],
        [52, 5.0, 48, 0, 42, 40, 45],
        [48, 4.5, 42, 0, 35, 34, 40]
    ], dtype=float)

    # --------------------------------------------------------
    # PERFORMANCE TARGET
    # --------------------------------------------------------

    performance_target = np.array([
        "Excellent",
        "Excellent",
        "Very Good",
        "Very Good",
        "Good",
        "Good",
        "Average",
        "Average",
        "Needs Improvement",
        "Needs Improvement",
        "Excellent",
        "Very Good",
        "Very Good",
        "Good",
        "Good",
        "Average",
        "Average",
        "Needs Improvement",
        "Needs Improvement",
        "Needs Improvement"
    ])

    # --------------------------------------------------------
    # PLACEMENT TARGET
    # --------------------------------------------------------

    placement_target = np.array([
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0
    ])

    # --------------------------------------------------------
    # PERFORMANCE MODEL
    # --------------------------------------------------------

    performance_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    performance_model.fit(
        X,
        performance_target
    )

    # --------------------------------------------------------
    # PLACEMENT MODEL
    # --------------------------------------------------------

    placement_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    placement_model.fit(
        X,
        placement_target
    )

    # --------------------------------------------------------
    # SAVE MODELS
    # --------------------------------------------------------

    with open(
        PERFORMANCE_MODEL,
        "wb"
    ) as file:

        pickle.dump(
            performance_model,
            file
        )

    with open(
        PLACEMENT_MODEL,
        "wb"
    ) as file:

        pickle.dump(
            placement_model,
            file
        )

    print(
        "AI models trained successfully."
    )

    print(
        f"Performance model: {PERFORMANCE_MODEL}"
    )

    print(
        f"Placement model: {PLACEMENT_MODEL}"
    )


# ============================================================
# RUN TRAINING
# ============================================================

if __name__ == "__main__":

    train_models()
