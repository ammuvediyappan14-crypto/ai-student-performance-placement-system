from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_from_directory,
    abort
)

import os
import pickle
import uuid
import mimetypes

import psycopg2
from psycopg2.extras import RealDictCursor

from pathlib import Path
from datetime import timedelta

import numpy as np

from werkzeug.utils import secure_filename


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

# Maximum profile image upload size: 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONSTANTS
# ============================================================

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


ALLOWED_DEPARTMENTS = {
    "CS",
    "IT",
    "AI",
    "BCA",
    "BBA",
    "B.COM"
}


ALLOWED_YEARS = {
    "I",
    "II",
    "III"
}


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

    connection = None

    try:

        connection = get_database()

        # ----------------------------------------------------
        # Create table if it does not exist
        # ----------------------------------------------------

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

                performance_score REAL,

                placement_probability REAL,

                placement_status TEXT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        connection.commit()

        # ----------------------------------------------------
        # IMPORTANT:
        # Existing PostgreSQL table may have been created
        # before profile_picture was added.
        #
        # CREATE TABLE IF NOT EXISTS does NOT modify an
        # existing table.
        #
        # Therefore check and add missing columns.
        # ----------------------------------------------------

        existing_columns_cursor = connection.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name = 'students'
            """
        )

        existing_columns = {
            row["column_name"]
            for row in existing_columns_cursor.fetchall()
        }

        required_columns = {

            "profile_picture": """
                ALTER TABLE students
                ADD COLUMN profile_picture TEXT
            """,

            "performance": """
                ALTER TABLE students
                ADD COLUMN performance TEXT
            """,

            "performance_score": """
                ALTER TABLE students
                ADD COLUMN performance_score REAL
            """,

            "placement_probability": """
                ALTER TABLE students
                ADD COLUMN placement_probability REAL
            """,

            "placement_status": """
                ALTER TABLE students
                ADD COLUMN placement_status TEXT
            """,

            "created_at": """
                ALTER TABLE students
                ADD COLUMN created_at
                TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            """
        }

        for column_name, alter_query in required_columns.items():

            if column_name not in existing_columns:

                try:

                    connection.execute(
                        alter_query
                    )

                    connection.commit()

                    print(
                        f"DATABASE MIGRATION: Added column '{column_name}'"
                    )

                except Exception as migration_error:

                    connection.rollback()

                    print(
                        f"DATABASE MIGRATION ERROR "
                        f"for '{column_name}':",
                        migration_error
                    )

        # ----------------------------------------------------
        # Final commit
        # ----------------------------------------------------

        connection.commit()

        print(
            "DATABASE INITIALIZATION SUCCESSFUL"
        )

    except Exception as error:

        if connection:

            try:
                connection.rollback()
            except Exception:
                pass

        print(
            "DATABASE INITIALIZATION ERROR:",
            error
        )

        raise

    finally:

        if connection:

            try:
                connection.close()
            except Exception:
                pass


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
# PROFILE PHOTO HELPERS
# ============================================================

def get_profile_extension(filename):

    if not filename:

        return None

    original_name = Path(
        filename
    ).name

    safe_name = secure_filename(
        original_name
    )

    if not safe_name:

        return None

    extension = Path(
        safe_name
    ).suffix.lower()

    if extension not in ALLOWED_IMAGE_EXTENSIONS:

        return None

    return extension


def save_profile_picture(file):

    """
    Save uploaded profile picture safely.

    Returns:
        generated filename or None
    """

    if not file:

        return None

    if not file.filename:

        return None

    extension = get_profile_extension(
        file.filename
    )

    if extension is None:

        raise ValueError(
            "Only JPG, JPEG, PNG and WEBP "
            "profile photos are allowed."
        )

    # --------------------------------------------------------
    # Generate unique filename.
    # --------------------------------------------------------

    generated_name = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    destination = (
        UPLOAD_FOLDER
        / generated_name
    )

    # --------------------------------------------------------
    # Save file.
    # --------------------------------------------------------

    try:

        file.save(
            str(destination)
        )

    except Exception as error:

        print(
            "PROFILE PHOTO SAVE ERROR:",
            error
        )

        raise ValueError(
            "Unable to save the profile photo."
        )

    # --------------------------------------------------------
    # Verify file really exists.
    # --------------------------------------------------------

    if not destination.exists():

        raise ValueError(
            "Profile photo upload failed."
        )

    return generated_name


def delete_profile_picture(filename):

    """
    Delete uploaded profile picture safely.
    """

    if not filename:

        return

    try:

        safe_name = secure_filename(
            Path(filename).name
        )

        if not safe_name:

            return

        file_path = (
            UPLOAD_FOLDER
            / safe_name
        )

        if file_path.exists():

            file_path.unlink()

    except Exception as error:

        print(
            "PROFILE PHOTO DELETE ERROR:",
            error
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

    connection = None

    try:

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

        return render_template(
            "dashboard.html",

            total_students=total_students,

            placed_students=placed_students,

            average_cgpa=round(
                float(average_cgpa or 0),
                2
            ),

            average_probability=round(
                float(average_probability or 0),
                2
            ),

            recent_students=recent_students
        )

    except Exception as error:

        print(
            "DASHBOARD ERROR:",
            error
        )

        flash(
            "Unable to load dashboard.",
            "danger"
        )

        return render_template(
            "dashboard.html",

            total_students=0,

            placed_students=0,

            average_cgpa=0,

            average_probability=0,

            recent_students=[]
        )

    finally:

        if connection:

            connection.close()


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

        uploaded_profile_name = None

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

            if department not in ALLOWED_DEPARTMENTS:

                raise ValueError(
                    "Please select a valid department."
                )

            if not year:

                raise ValueError(
                    "Year is required."
                )

            if year not in ALLOWED_YEARS:

                raise ValueError(
                    "Please select a valid year."
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

            if (
                profile_picture is None
                or not profile_picture.filename
            ):

                # No photo selected.
                profile_picture_name = None

            else:

                profile_picture_name = (
                    save_profile_picture(
                        profile_picture
                    )
                )

                uploaded_profile_name = (
                    profile_picture_name
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
                    performance_score,
                    placement_probability,
                    placement_status

                )

                VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?, ?
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

                    float(performance)
                    if str(performance).replace(
                        ".",
                        "",
                        1
                    ).isdigit()
                    else None,

                    probability,

                    status
                )
            )

            student_row = cursor.fetchone()

            if not student_row:

                raise RuntimeError(
                    "Student ID was not returned by database."
                )

            student_id = student_row["id"]

            connection.commit()

            connection.close()

            connection = None

            # Photo has been successfully associated
            # with the database record.
            uploaded_profile_name = None

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

        except psycopg2.IntegrityError as error:

            print(
                "STUDENT DATABASE INTEGRITY ERROR:",
                error
            )

            if connection:

                try:
                    connection.rollback()
                except Exception:
                    pass

                try:
                    connection.close()
                except Exception:
                    pass

                connection = None

            # Remove photo because student was not inserted.
            if uploaded_profile_name:

                delete_profile_picture(
                    uploaded_profile_name
                )

            flash(
                "This roll number already exists.",
                "danger"
            )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        except ValueError as error:

            print(
                "STUDENT VALIDATION ERROR:",
                error
            )

            if connection:

                try:
                    connection.rollback()
                except Exception:
                    pass

                try:
                    connection.close()
                except Exception:
                    pass

                connection = None

            if uploaded_profile_name:

                delete_profile_picture(
                    uploaded_profile_name
                )

            flash(
                str(error),
                "danger"
            )

        # ----------------------------------------------------
        # OTHER ERROR
        # ----------------------------------------------------

        except Exception as error:

            print(
                "STUDENT SAVE ERROR:",
                error
            )

            if connection:

                try:
                    connection.rollback()
                except Exception:
                    pass

                try:
                    connection.close()
                except Exception:
                    pass

                connection = None

            if uploaded_profile_name:

                delete_profile_picture(
                    uploaded_profile_name
                )

            flash(
                "Unable to add student. "
                "Please check the entered details.",
                "danger"
            )

    # ========================================================
    # GET
    # ========================================================

    connection = None

    try:

        connection = get_database()

        student_list = connection.execute(
            """
            SELECT *
            FROM students
            ORDER BY id DESC
            """
        ).fetchall()

    except Exception as error:

        print(
            "STUDENT LIST ERROR:",
            error
        )

        flash(
            "Unable to load student records.",
            "danger"
        )

        student_list = []

    finally:

        if connection:

            connection.close()

    return render_template(
        "students.html",
        students=student_list
    )


# ============================================================
# PROFILE PHOTO ROUTE
# ============================================================

@app.route(
    "/uploads/<path:filename>"
)
def uploaded_file(filename):

    # --------------------------------------------------------
    # Only logged-in administrator can view uploaded photos.
    # --------------------------------------------------------

    if not session.get("admin"):

        return redirect(
            url_for("login")
        )

    # --------------------------------------------------------
    # Prevent unsafe path access.
    # --------------------------------------------------------

    safe_name = secure_filename(
        Path(filename).name
    )

    if not safe_name:

        abort(404)

    file_path = (
        UPLOAD_FOLDER
        / safe_name
    )

    if not file_path.exists():

        abort(404)

    return send_from_directory(
        str(UPLOAD_FOLDER),
        safe_name
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

    connection = None

    try:

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

    except Exception as error:

        print(
            "STUDENT DETAIL ERROR:",
            error
        )

        flash(
            "Unable to load student details.",
            "danger"
        )

        return redirect(
            url_for("students")
        )

    finally:

        if connection:

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

    connection = None

    profile_picture_name = None

    try:

        connection = get_database()

        # ----------------------------------------------------
        # Get profile picture before deleting DB record.
        # ----------------------------------------------------

        student = connection.execute(
            """
            SELECT profile_picture
            FROM students
            WHERE id = ?
            """,
            (
                student_id,
            )
        ).fetchone()

        if student:

            profile_picture_name = (
                student["profile_picture"]
            )

        # ----------------------------------------------------
        # Delete student.
        # ----------------------------------------------------

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

        connection = None

        # ----------------------------------------------------
        # Delete associated profile photo.
        # ----------------------------------------------------

        if profile_picture_name:

            delete_profile_picture(
                profile_picture_name
            )

        flash(
            "Student deleted successfully.",
            "success"
        )

    except Exception as error:

        print(
            "DELETE STUDENT ERROR:",
            error
        )

        if connection:

            try:
                connection.rollback()
            except Exception:
                pass

            try:
                connection.close()
            except Exception:
                pass

        flash(
            "Unable to delete student.",
            "danger"
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
# MAXIMUM FILE SIZE ERROR
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    flash(
        "Profile photo is too large. "
        "Maximum allowed size is 5 MB.",
        "danger"
    )

    return redirect(
        url_for("students")
    )


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
