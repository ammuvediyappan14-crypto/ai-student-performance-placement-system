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
import psycopg2
from psycopg2.extras import RealDictCursor
from pathlib import Path
from datetime import timedelta

import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

PERFORMANCE_MODEL = MODEL_DIR / "performance_model.pkl"

PLACEMENT_MODEL = MODEL_DIR / "placement_model.pkl"

UPLOAD_FOLDER = BASE_DIR / "uploads"


# ============================================================
# DATABASE URL
# ============================================================

DATABASE_URL = os.environ.get("DATABASE_URL")


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "ai-student-performance-secret-key"
)

app.config["SESSION_COOKIE_NAME"] = (
    "student_system_session_v3"
)

app.config["SESSION_COOKIE_HTTPONLY"] = True

app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

app.config["SESSION_COOKIE_SECURE"] = False

app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(
    hours=12
)

app.config["UPLOAD_FOLDER"] = str(
    UPLOAD_FOLDER
)

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
# POSTGRESQL DATABASE CONNECTION
# ============================================================

class DatabaseConnection:

    def __init__(self):

        if not DATABASE_URL:

            raise RuntimeError(
                "DATABASE_URL environment variable is not configured."
            )

        self.connection = psycopg2.connect(
            DATABASE_URL,
            cursor_factory=RealDictCursor
        )

    def execute(self, query, params=None):

        # Convert SQLite-style placeholders
        # to PostgreSQL placeholders.

        query = query.replace(
            "?",
            "%s"
        )

        cursor = self.connection.cursor()

        cursor.execute(
            query,
            params or ()
        )

        return cursor

    def commit(self):

        self.connection.commit()

    def rollback(self):

        self.connection.rollback()

    def close(self):

        self.connection.close()


def get_database():

    return DatabaseConnection()


# ============================================================
# DATABASE INITIALIZATION
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

            id SERIAL PRIMARY KEY,

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


# ============================================================
# LOAD MODELS
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


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    if session.get("admin"):

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

        if (
            username == "admin"
            and password == "admin123"
        ):

            session.clear()

            session["admin"] = "admin"

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

    if not session.get("admin"):

        return redirect(
            url_for("login")
        )

    connection = get_database()

    total_students = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM students
        """
    ).fetchone()["count"]

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

    average_cgpa = connection.execute(
        """
        SELECT AVG(cgpa) AS average
        FROM students
        """
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
        LIMIT 10
        """
    ).fetchall()

    connection.close()

    return render_template(
        "dashboard.html",

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
# GET  -> FORM + EXISTING STUDENTS
# POST -> SAVE STUDENT + AI PREDICTION
# ============================================================

@app.route(
    "/students",
    methods=["GET", "POST"]
)
def students():

    if not session.get("admin"):

        return redirect(
            url_for("login")
        )

    # ========================================================
    # POST
    # ========================================================

    if request.method == "POST":

        connection = None

        try:

            # ------------------------------------------------
            # BASIC DETAILS
            # ------------------------------------------------

            roll_no = request.form.get(
                "roll_no",
                ""
            ).strip()

            name = request.form.get(
                "name",
                ""
            ).strip()

            department = request.form.get(
                "department",
                ""
            ).strip()

            year = request.form.get(
                "year",
                ""
            ).strip()

            # ------------------------------------------------
            # REQUIRED FIELD CHECK
            # ------------------------------------------------

            if not roll_no:

                raise ValueError(
                    "Roll number is required."
                )

            if not name:

                raise ValueError(
                    "Student name is required."
                )

            if not department:

                raise ValueError(
                    "Department is required."
                )

            if not year:

                raise ValueError(
                    "Year is required."
                )

            # ------------------------------------------------
            # NUMERIC VALUES
            # ------------------------------------------------

            values = {

                "attendance": float(
                    request.form.get(
                        "attendance",
                        0
                    )
                ),

                "cgpa": float(
                    request.form.get(
                        "cgpa",
                        0
                    )
                ),

                "internal_marks": float(
                    request.form.get(
                        "internal_marks",
                        0
                    )
                ),

                "projects": int(
                    request.form.get(
                        "projects",
                        0
                    )
                ),

                "skills_score": float(
                    request.form.get(
                        "skills_score",
                        0
                    )
                ),

                "aptitude_score": float(
                    request.form.get(
                        "aptitude_score",
                        0
                    )
                ),

                "communication_score": float(
                    request.form.get(
                        "communication_score",
                        0
                    )
                )
            }

            # ------------------------------------------------
            # RANGE VALIDATION
            # ------------------------------------------------

            ranges = [

                (
                    "Attendance",
                    values["attendance"],
                    0,
                    100
                ),

                (
                    "CGPA",
                    values["cgpa"],
                    0,
                    10
                ),

                (
                    "Internal marks",
                    values["internal_marks"],
                    0,
                    100
                ),

                (
                    "Projects",
                    values["projects"],
                    0,
                    20
                ),

                (
                    "Skills score",
                    values["skills_score"],
                    0,
                    100
                ),

                (
                    "Aptitude score",
                    values["aptitude_score"],
                    0,
                    100
                ),

                (
                    "Communication score",
                    values["communication_score"],
                    0,
                    100
                )
            ]

            for (
                field_name,
                value,
                minimum,
                maximum
            ) in ranges:

                if not (
                    minimum
                    <= value
                    <= maximum
                ):

                    raise ValueError(
                        f"{field_name} must be between "
                        f"{minimum} and {maximum}."
                    )

            # ------------------------------------------------
            # PROFILE PHOTO
            # ------------------------------------------------

            profile_picture = request.files.get(
                "profile_picture"
            )

            profile_picture_name = None

            if (
                profile_picture
                and profile_picture.filename
            ):

                original_name = Path(
                    profile_picture.filename
                ).name

                extension = Path(
                    original_name
                ).suffix.lower()

                allowed_extensions = {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp"
                }

                if extension not in allowed_extensions:

                    raise ValueError(
                        "Only JPG, JPEG, PNG and WEBP "
                        "profile photos are allowed."
                    )

                import uuid

                profile_picture_name = (
                    str(uuid.uuid4())
                    + extension
                )

                profile_picture.save(
                    UPLOAD_FOLDER
                    / profile_picture_name
                )

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
            # SAVE STUDENT TO POSTGRESQL
            # ------------------------------------------------

            connection = get_database()

            cursor = connection.execute(
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

                RETURNING id
                """,
                (

                    roll_no,

                    name,

                    department,

                    profile_picture_name,

                    year,

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

            # PostgreSQL returns inserted ID
            student_id = cursor.fetchone()["id"]

            connection.commit()

            connection.close()

            connection = None

            flash(
                "Student added successfully. "
                "AI performance and placement prediction generated.",
                "success"
            )

            # ------------------------------------------------
            # OPEN SAVED STUDENT DETAILS
            # ------------------------------------------------

            return redirect(
                url_for(
                    "student_detail",
                    student_id=student_id
                )
            )

        # ----------------------------------------------------
        # DUPLICATE ROLL NUMBER
        # ----------------------------------------------------

        except psycopg2.IntegrityError:

            if connection:

                connection.rollback()
                connection.close()

            flash(
                "This roll number already exists.",
                "danger"
            )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        except ValueError as error:

            if connection:

                connection.rollback()
                connection.close()

            flash(
                str(error),
                "danger"
            )

        # ----------------------------------------------------
        # OTHER ERROR
        # ----------------------------------------------------

        except Exception as error:

            if connection:

                connection.rollback()
                connection.close()

            print(
                "STUDENT SAVE ERROR:",
                error
            )

            flash(
                "Unable to add student. "
                "Please check the entered details.",
                "danger"
            )

    # ========================================================
    # GET
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
# PROFILE PHOTO
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
# STUDENT DETAILS / PREDICTION
# ============================================================

@app.route(
    "/student/<int:student_id>"
)
def student_detail(student_id):

    if not session.get("admin"):

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

    if not session.get("admin"):

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
