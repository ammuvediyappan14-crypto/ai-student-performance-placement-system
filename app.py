from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)

import os
import pickle
import sqlite3

from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent

DATABASE = BASE_DIR / "student_system.db"

MODEL_DIR = BASE_DIR / "models"

PERFORMANCE_MODEL = MODEL_DIR / "performance_model.pkl"
PLACEMENT_MODEL = MODEL_DIR / "placement_model.pkl"


app = Flask(__name__) 
UPLOAD_FOLDER = BASE_DIR / "uploads"
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "development-secret-key"
)


FEATURES = [
    "attendance",
    "cgpa",
    "internal_marks",
    "projects",
    "skills_score",
    "aptitude_score",
    "communication_score"
]


def get_database():
    connection = sqlite3.connect(DATABASE)

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():
    MODEL_DIR.mkdir(exist_ok=True)

    connection = get_database()

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            roll_no TEXT UNIQUE NOT NULL,

            name TEXT NOT NULL,

            department TEXT NOT NULL,
            
            profile_picture TEXT,

            year TEXT NOT NULL,

            attendance REAL NOT NULL,

            cgpa REAL NOT NULL,

            internal_marks REAL NOT NULL,

            projects INTEGER NOT NULL,

            skills_score REAL NOT NULL,

            aptitude_score REAL NOT NULL,

            communication_score REAL NOT NULL,

            performance TEXT,

            placement_probability REAL,

            placement_status TEXT
        )
        """
    )

    connection.commit()

    connection.close()


def load_models():

    if (
        not PERFORMANCE_MODEL.exists()
        or not PLACEMENT_MODEL.exists()
    ):

        from train_model import train_models

        train_models()

    with open(
        PERFORMANCE_MODEL,
        "rb"
    ) as file:

        performance_model = pickle.load(file)

    with open(
        PLACEMENT_MODEL,
        "rb"
    ) as file:

        placement_model = pickle.load(file)

    return performance_model, placement_model


def predict_student(values):

    performance_model, placement_model = load_models()

    input_data = np.array(
        [[values[feature] for feature in FEATURES]],
        dtype=float
    )

    performance = performance_model.predict(
        input_data
    )[0]

    probability = (
        placement_model.predict_proba(
            input_data
        )[0][1] * 100
    )

    probability = round(
        float(probability),
        2
    )

    if probability >= 60:

        status = "Likely to be Placed"

    else:

        status = "Needs Improvement"

    return (
        str(performance),
        probability,
        status
    )


@app.route("/")
def home():

    if "admin" in session:

        return redirect(
            url_for("dashboard")
        )

    return redirect(
        url_for("login")
    )


@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        if (
            username == "admin"
            and password == "admin123"
        ):

            session["admin"] = username

            return redirect(
                url_for("dashboard")
            )

        flash(
            "Invalid username or password.",
            "danger"
        )

    return render_template(
        "login.html"
    )


@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


@app.route("/dashboard")
def dashboard():

    if "admin" not in session:

        return redirect(
            url_for("login")
        )

    connection = get_database()

    total_students = connection.execute(
        "SELECT COUNT(*) AS count FROM students"
    ).fetchone()["count"]

    placed_students = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM students
        WHERE placement_status = ?
        """,
        ("Likely to be Placed",)
    ).fetchone()["count"]

    average_cgpa = connection.execute(
        "SELECT AVG(cgpa) AS average FROM students"
    ).fetchone()["average"]

    average_probability = connection.execute(
        """
        SELECT AVG(placement_probability) AS average
        FROM students
        """
    ).fetchone()["average"]

    recent_students = connection.execute(
        """
        SELECT *
        FROM students
        ORDER BY id DESC
        LIMIT 50
        """
    ).fetchall()

    connection.close()

    return render_template(
        "dashboard.html",
        total=total_students,
        placed=placed_students,
        avg_cgpa=round(
            average_cgpa or 0,
            2
        ),
        avg_probability=round(
            average_probability or 0,
            2
        ),
        recent=recent_students
    )


@app.route(
    "/students",
    methods=["GET", "POST"]
)
def students():

    if "admin" not in session:

        return redirect(
            url_for("login")
        )

    if request.method == "POST":

        try:
            profile_picture = request.files.get("profile_picture")

            if profile_picture and profile_picture.filename:
            profile_picture.save(
                UPLOAD_FOLDER / profile_picture.filename
            )

            values = {

                "attendance":
                    float(
                        request.form[
                            "attendance"
                        ]
                    ),

                "cgpa":
                    float(
                        request.form[
                            "cgpa"
                        ]
                    ),

                "internal_marks":
                    float(
                        request.form[
                            "internal_marks"
                        ]
                    ),

                "projects":
                    int(
                        request.form[
                            "projects"
                        ]
                    ),

                "skills_score":
                    float(
                        request.form[
                            "skills_score"
                        ]
                    ),

                "aptitude_score":
                    float(
                        request.form[
                            "aptitude_score"
                        ]
                    ),

                "communication_score":
                    float(
                        request.form[
                            "communication_score"
                        ]
                    )
            }

            ranges = [

                (
                    values["attendance"],
                    0,
                    100
                ),

                (
                    values["cgpa"],
                    0,
                    10
                ),

                (
                    values["internal_marks"],
                    0,
                    100
                ),

                (
                    values["projects"],
                    0,
                    20
                ),

                (
                    values["skills_score"],
                    0,
                    100
                ),

                (
                    values["aptitude_score"],
                    0,
                    100
                ),

                (
                    values["communication_score"],
                    0,
                    100
                )
            ]

            for value, minimum, maximum in ranges:

                if not (
                    minimum
                    <= value
                    <= maximum
                ):

                    raise ValueError

            performance, probability, status = (
                predict_student(values)
            )

            connection = get_database()

            connection.execute(
                """
                INSERT INTO students (
                    roll_no,
                    name,
                    department,
                    profile_picture,
                    year,
                    attendance,
                    cgpa,
                    internal_marks,
                    projects,
                    skills_score,
                    aptitude_score,
                    communication_score,
                    performance,
                    placement_probability,
                    placement_status
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    request.form[
                        "roll_no"
                    ].strip(),

                    request.form[
                        "name"
                    ].strip(),

                    request.form[
                        "department"
                    ].strip(),
                    profile_picture.filename if profile_picture else None,

                    request.form[
                        "year"
                    ].strip(),

                    values["attendance"],

                    values["cgpa"],

                    values["internal_marks"],

                    values["projects"],

                    values["skills_score"],

                    values["aptitude_score"],

                    values["communication_score"],

                    performance,

                    probability,

                    status
                )
            )

            connection.commit()

            connection.close()

            flash(
                "Student added successfully and AI prediction generated.",
                "success"
            )

            return redirect(
                url_for("students")
            )

        except sqlite3.IntegrityError:

            flash(
                "Roll number already exists.",
                "danger"
            )

        except (
            KeyError,
            ValueError
        ):

            flash(
                "Please enter valid values.",
                "danger"
            )

    connection = get_database()

    student_list = connection.execute(
        """
        SELECT *
        FROM students
        ORDER BY id DESC
        """
    ).fetchall()

    connection.close()

    return render_template(
        "students.html",
        students=student_list
    )


@app.route(
    "/student/<int:student_id>"
)
def student_detail(student_id):

    if "admin" not in session:

        return redirect(
            url_for("login")
        )

    connection = get_database()

    student = connection.execute(
        """
        SELECT *
        FROM students
        WHERE id = ?
        """,
        (student_id,)
    ).fetchone()

    connection.close()

    if student is None:

        flash(
            "Student not found.",
            "danger"
        )

        return redirect(
            url_for("students")
        )

    return render_template(
        "prediction.html",
        student=student
    )


@app.route(
    "/students/delete/<int:student_id>",
    methods=["POST"]
)
def delete_student(student_id):

    if "admin" not in session:

        return redirect(
            url_for("login")
        )

    connection = get_database()

    connection.execute(
        """
        DELETE FROM students
        WHERE id = ?
        """,
        (student_id,)
    )

    connection.commit()

    connection.close()

    flash(
        "Student deleted successfully.",
        "success"
    )

    return redirect(
        url_for("students")
    )


@app.route("/health")
def health():

    return {
        "status": "ok"
    }


# IMPORTANT:
# Initialize database when Flask is imported by Gunicorn.
initialize_database()


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=True
    )
