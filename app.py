from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_from_directory
)

import os
import pickle
import sqlite3
from pathlib import Path
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE = BASE_DIR / "student_system.db"
MODEL_DIR = BASE_DIR / "models"

PERFORMANCE_MODEL = MODEL_DIR / "performance_model.pkl"
PLACEMENT_MODEL = MODEL_DIR / "placement_model.pkl"

UPLOAD_FOLDER = BASE_DIR / "uploads"


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "development-secret-key"
)

# Session settings
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Upload settings
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ML FEATURES
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
# DATABASE CONNECTION
# ============================================================

def get_database():

    connection = sqlite3.connect(
        DATABASE
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    UPLOAD_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

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

    # --------------------------------------------------------
    # Check existing database columns
    # --------------------------------------------------------

    columns = connection.execute(
        "PRAGMA table_info(students)"
    ).fetchall()

    column_names = [
        column["name"]
        for column in columns
    ]

    # --------------------------------------------------------
    # Add profile_picture if old database doesn't have it
    # --------------------------------------------------------

    if "profile_picture" not in column_names:

        connection.execute(
            """
            ALTER TABLE students
            ADD COLUMN profile_picture TEXT
            """
        )

    connection.commit()

    connection.close()


# ============================================================
# LOAD / TRAIN ML MODELS
# ============================================================

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

        performance_model = pickle.load(
            file
        )

    with open(
        PLACEMENT_MODEL,
        "rb"
    ) as file:

        placement_model = pickle.load(
            file
        )

    return (
        performance_model,
        placement_model
    )


# ============================================================
# AI PREDICTION
# ============================================================

def predict_student(values):

    (
        performance_model,
        placement_model
    ) = load_models()

    input_data = np.array(
        [
            [
                values[feature]
                for feature in FEATURES
            ]
        ],
        dtype=float
    )

    performance = performance_model.predict(
        input_data
    )[0]

    probability = (
        placement_model
        .predict_proba(input_data)[0][1]
        * 100
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


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    if "admin" in session:

        return redirect(
            url_for("dashboard")
        )

    return redirect(
        url_for("login")
    )


# ============================================================
# LOGIN
# ============================================================

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

        # ----------------------------------------------------
        # ADMIN LOGIN
        # ----------------------------------------------------

        if (
            username == "admin"
            and password == "admin123"
        ):

            session.clear()

            session["admin"] = username

            session.permanent = True

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


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    if "admin" not in session:

        return redirect(
            url_for("login")
        )

    connection = get_database()

    # --------------------------------------------------------
    # Total students
    # --------------------------------------------------------

    total_students = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM students
        """
    ).fetchone()["count"]

    # --------------------------------------------------------
    # Placed students
    # --------------------------------------------------------

    placed_students = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM students
        WHERE placement_status = ?
        """,
        (
            "Likely to be Placed",
        )
    ).fetchone()["count"]

    # --------------------------------------------------------
    # Average CGPA
    # --------------------------------------------------------

    average_cgpa = connection.execute(
        """
        SELECT AVG(cgpa) AS average
        FROM students
        """
    ).fetchone()["average"]

    # --------------------------------------------------------
    # Average placement probability
    # --------------------------------------------------------

    average_probability = connection.execute(
        """
        SELECT AVG(placement_probability) AS average
        FROM students
        """
    ).fetchone()["average"]

    # --------------------------------------------------------
    # Recent students
    # --------------------------------------------------------

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

        # Names used by dashboard.html
        total_students=total_students,

        placed_students=placed_students,

        average_cgpa=round(
            average_cgpa or 0,
            2
        ),

        average_probability=round(
            average_probability or 0,
            2
        ),

        recent_students=recent_students
    )


# ============================================================
# STUDENTS
# ============================================================

@app.route(
    "/students",
    methods=["GET", "POST"]
)
def students():

    # --------------------------------------------------------
    # LOGIN CHECK
    # --------------------------------------------------------

    if "admin" not in session:

        return redirect(
            url_for("login")
        )

    # ========================================================
    # POST - ADD STUDENT
    # ========================================================

    if request.method == "POST":

        try:

            # ------------------------------------------------
            # PROFILE PICTURE
            # ------------------------------------------------

            profile_picture = request.files.get(
                "profile_picture"
            )

            profile_picture_name = None

            if (
                profile_picture
                and profile_picture.filename
            ):

                filename = Path(
                    profile_picture.filename
                ).name

                profile_picture.save(
                    UPLOAD_FOLDER / filename
                )

                profile_picture_name = filename

            # ------------------------------------------------
            # STUDENT VALUES
            # ------------------------------------------------

            values = {

                "attendance": float(
                    request.form["attendance"]
                ),

                "cgpa": float(
                    request.form["cgpa"]
                ),

                "internal_marks": float(
                    request.form["internal_marks"]
                ),

                "projects": int(
                    request.form["projects"]
                ),

                "skills_score": float(
                    request.form["skills_score"]
                ),

                "aptitude_score": float(
                    request.form["aptitude_score"]
                ),

                "communication_score": float(
                    request.form["communication_score"]
                )
            }

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

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

            for (
                value,
                minimum,
                maximum
            ) in ranges:

                if not (
                    minimum
                    <= value
                    <= maximum
                ):

                    raise ValueError

            # ------------------------------------------------
            # AI PREDICTION
            # ------------------------------------------------

            (
                performance,
                probability,
                status
            ) = predict_student(
                values
            )

            # ------------------------------------------------
            # DATABASE INSERT
            # ------------------------------------------------

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
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?
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

                    profile_picture_name,

                    request.form[
                        "year"
                    ].strip(),

                    values[
                        "attendance"
                    ],

                    values[
                        "cgpa"
                    ],

                    values[
                        "internal_marks"
                    ],

                    values[
                        "projects"
                    ],

                    values[
                        "skills_score"
                    ],

                    values[
                        "aptitude_score"
                    ],

                    values[
                        "communication_score"
                    ],

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

        # ----------------------------------------------------
        # DUPLICATE ROLL NUMBER
        # ----------------------------------------------------

        except sqlite3.IntegrityError:

            flash(
                "Roll number already exists.",
                "danger"
            )

        # ----------------------------------------------------
        # INVALID VALUES
        # ----------------------------------------------------

        except (
            KeyError,
            ValueError
        ):

            flash(
                "Please enter valid values.",
                "danger"
            )

        # ----------------------------------------------------
        # OTHER ERROR
        # ----------------------------------------------------

        except Exception as error:

            print(
                "STUDENT ERROR:",
                error
            )

            flash(
                "Unable to add student. Please check the entered details.",
                "danger"
            )

    # ========================================================
    # GET - SHOW STUDENTS
    # ========================================================

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


# ============================================================
# UPLOADED PROFILE PICTURES
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# STUDENT DETAILS / AI PREDICTION
# ============================================================

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
        (
            student_id,
        )
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


# ============================================================
# DELETE STUDENT
# ============================================================

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
        (
            student_id,
        )
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


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "ok"
    }


# ============================================================
# TEST ROUTE
# ============================================================

@app.route("/test-students")
def test_students():

    return "STUDENTS ROUTE IS WORKING"


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

initialize_database()


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

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
