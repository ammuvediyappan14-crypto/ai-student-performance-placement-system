import os
import pickle
from pathlib import Path
from io import BytesIO

import numpy as np
import psycopg2
from psycopg2 import Binary
from psycopg2.extras import RealDictCursor

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
    send_file,
)

from werkzeug.utils import secure_filename


# =========================================================
# APP CONFIGURATION
# =========================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "student-performance-system-secret-key"
)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

DATABASE_URL = os.environ.get("DATABASE_URL")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

LEGACY_UPLOAD_FOLDER = BASE_DIR / "uploads"


# =========================================================
# ALLOWED PROFILE IMAGE TYPES
# =========================================================

ALLOWED_EXTENSIONS = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}


# =========================================================
# DATABASE CONNECTION
# =========================================================

class DatabaseConnection:

    def __init__(self):
        self.connection = None

    def connect(self):

        if not DATABASE_URL:
            raise RuntimeError(
                "DATABASE_URL environment variable is not configured."
            )

        self.connection = psycopg2.connect(
            DATABASE_URL,
            sslmode="require"
        )

        return self.connection

    def execute(
        self,
        query,
        params=None,
        fetch=False,
        fetchone=False
    ):

        self.connect()

        try:

            cursor = self.connection.cursor(
                cursor_factory=RealDictCursor
            )

            cursor.execute(
                query,
                params or ()
            )

            result = None

            if fetchone:
                result = cursor.fetchone()

            elif fetch:
                result = cursor.fetchall()

            self.connection.commit()

            cursor.close()

            return result

        except Exception:

            self.connection.rollback()

            raise

        finally:

            self.connection.close()


# =========================================================
# DATABASE INITIALIZATION
# =========================================================

def initialize_database():

    db = DatabaseConnection()

    create_table_query = """
    CREATE TABLE IF NOT EXISTS students (
        id SERIAL PRIMARY KEY,
        roll_no TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        department TEXT NOT NULL,
        profile_picture TEXT,
        profile_image BYTEA,
        profile_image_mimetype TEXT,
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

    db.execute(create_table_query)

    # -----------------------------------------------------
    # MIGRATION FOR EXISTING DATABASE
    # -----------------------------------------------------

    migration_columns = {
        "profile_picture": "TEXT",
        "profile_image": "BYTEA",
        "profile_image_mimetype": "TEXT",
        "performance_score": "REAL",
        "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    }

    for column_name, column_type in migration_columns.items():

        check_query = """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = 'students'
            AND column_name = %s
        ) AS exists;
        """

        result = db.execute(
            check_query,
            (column_name,),
            fetchone=True
        )

        if not result["exists"]:

            alter_query = f"""
            ALTER TABLE students
            ADD COLUMN {column_name} {column_type}
            """

            db.execute(alter_query)


# =========================================================
# MODEL LOADING
# =========================================================

performance_model = None
placement_model = None


def load_models():

    global performance_model
    global placement_model

    performance_model = None
    placement_model = None

    performance_model_path = (
        MODELS_DIR / "performance_model.pkl"
    )

    placement_model_path = (
        MODELS_DIR / "placement_model.pkl"
    )

    try:

        if performance_model_path.exists():

            with open(
                performance_model_path,
                "rb"
            ) as file:

                performance_model = pickle.load(file)

    except Exception as error:

        print(
            "Performance model loading failed:",
            error
        )

    try:

        if placement_model_path.exists():

            with open(
                placement_model_path,
                "rb"
            ) as file:

                placement_model = pickle.load(file)

    except Exception as error:

        print(
            "Placement model loading failed:",
            error
        )


# =========================================================
# AI PREDICTION
# =========================================================

FEATURES = [
    "attendance",
    "cgpa",
    "internal_marks",
    "projects",
    "skills_score",
    "aptitude_score",
    "communication_score",
]


def calculate_fallback_prediction(data):

    attendance = float(data["attendance"])
    cgpa = float(data["cgpa"])
    internal_marks = float(data["internal_marks"])
    projects = float(data["projects"])
    skills_score = float(data["skills_score"])
    aptitude_score = float(data["aptitude_score"])
    communication_score = float(
        data["communication_score"]
    )

    cgpa_score = cgpa * 10

    project_score = min(
        projects / 20 * 100,
        100
    )

    score = (
        attendance * 0.15
        + cgpa_score * 0.20
        + internal_marks * 0.15
        + project_score * 0.10
        + skills_score * 0.15
        + aptitude_score * 0.10
        + communication_score * 0.15
    )

    performance_score = round(
        max(
            0,
            min(score, 100)
        ),
        2
    )

    if performance_score >= 75:
        performance = "Excellent"

    elif performance_score >= 60:
        performance = "Good"

    elif performance_score >= 45:
        performance = "Average"

    else:
        performance = "Needs Improvement"

    placement_probability = round(
        max(
            0,
            min(score, 100)
        ),
        2
    )

    if placement_probability >= 75:
        placement_status = "High"

    elif placement_probability >= 50:
        placement_status = "Medium"

    else:
        placement_status = "Low"

    return (
        performance,
        performance_score,
        placement_probability,
        placement_status,
    )


def predict_student(data):

    features = np.array([
        [
            float(data["attendance"]),
            float(data["cgpa"]),
            float(data["internal_marks"]),
            float(data["projects"]),
            float(data["skills_score"]),
            float(data["aptitude_score"]),
            float(data["communication_score"]),
        ]
    ])

    try:

        if performance_model is not None:

            performance_prediction = (
                performance_model.predict(
                    features
                )[0]
            )

            performance = str(
                performance_prediction
            )

        else:

            performance = None

        if placement_model is not None:

            if hasattr(
                placement_model,
                "predict_proba"
            ):

                probabilities = (
                    placement_model.predict_proba(
                        features
                    )[0]
                )

                if len(probabilities) > 1:

                    placement_probability = (
                        float(
                            probabilities[-1]
                        ) * 100
                    )

                else:

                    placement_probability = (
                        float(
                            probabilities[0]
                        ) * 100
                    )

            else:

                prediction = (
                    placement_model.predict(
                        features
                    )[0]
                )

                placement_probability = (
                    float(prediction) * 100
                )

            placement_probability = round(
                max(
                    0,
                    min(
                        placement_probability,
                        100
                    )
                ),
                2
            )

        else:

            raise Exception(
                "Placement model unavailable"
            )

        performance_score = round(
            placement_probability,
            2
        )

        if placement_probability >= 75:

            placement_status = "High"

        elif placement_probability >= 50:

            placement_status = "Medium"

        else:

            placement_status = "Low"

        return (
            performance,
            performance_score,
            placement_probability,
            placement_status,
        )

    except Exception as error:

        print(
            "Model prediction fallback:",
            error
        )

        return calculate_fallback_prediction(
            data
        )


# =========================================================
# IMAGE HELPERS
# =========================================================

def allowed_image(filename):

    if not filename:
        return False

    filename = secure_filename(
        filename
    )

    if "." not in filename:
        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


def get_image_mimetype(filename):

    filename = secure_filename(
        filename
    )

    if "." not in filename:
        return "application/octet-stream"

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return ALLOWED_EXTENSIONS.get(
        extension,
        "application/octet-stream"
    )


# =========================================================
# LEGACY PHOTO RECOVERY
# =========================================================

def find_legacy_profile_image(profile_picture):

    if not profile_picture:
        return None, None

    filename = secure_filename(
        str(profile_picture)
    )

    if not filename:
        return None, None

    legacy_file = (
        LEGACY_UPLOAD_FOLDER / filename
    )

    if not legacy_file.exists():
        return None, None

    if not legacy_file.is_file():
        return None, None

    try:

        image_data = legacy_file.read_bytes()

    except Exception as error:

        print(
            "Legacy profile image read failed:",
            error
        )

        return None, None

    if not image_data:
        return None, None

    mimetype = get_image_mimetype(
        filename
    )

    return image_data, mimetype


def migrate_legacy_profile_image(
    student_id,
    profile_picture
):

    image_data, mimetype = (
        find_legacy_profile_image(
            profile_picture
        )
    )

    if not image_data:
        return False

    db = DatabaseConnection()

    try:

        db.execute(
            """
            UPDATE students
            SET
                profile_image = %s,
                profile_image_mimetype = %s
            WHERE id = %s
            """,
            (
                Binary(image_data),
                mimetype,
                student_id,
            )
        )

        print(
            f"Legacy profile image migrated for student {student_id}."
        )

        return True

    except Exception as error:

        print(
            "Legacy profile image migration failed:",
            error
        )

        return False


# =========================================================
# LOGIN
# =========================================================

@app.route("/")
def home():

    if session.get("admin"):

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

            session.clear()

            session["admin"] = True

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


@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    if not session.get("admin"):

        return redirect(
            url_for("login")
        )

    db = DatabaseConnection()

    total_result = db.execute(
        """
        SELECT COUNT(*) AS count
        FROM students
        """,
        fetchone=True
    )

    total_students = (
        total_result["count"]
        if total_result
        else 0
    )

    placed_result = db.execute(
        """
        SELECT COUNT(*) AS count
        FROM students
        WHERE placement_probability >= 50
        """,
        fetchone=True
    )

    placed_students = (
        placed_result["count"]
        if placed_result
        else 0
    )

    average_cgpa_result = db.execute(
        """
        SELECT AVG(cgpa) AS average
        FROM students
        """,
        fetchone=True
    )

    average_cgpa = (
        round(
            float(
                average_cgpa_result["average"]
            ),
            2
        )
        if (
            average_cgpa_result
            and average_cgpa_result["average"]
            is not None
        )
        else 0
    )

    average_probability_result = (
        db.execute(
            """
            SELECT AVG(placement_probability) AS average
            FROM students
            """,
            fetchone=True
        )
    )

    average_probability = (
        round(
            float(
                average_probability_result["average"]
            ),
            2
        )
        if (
            average_probability_result
            and average_probability_result["average"]
            is not None
        )
        else 0
    )

    recent_students = db.execute(
        """
        SELECT
            id,
            roll_no,
            name,
            department,
            year,
            cgpa,
            placement_probability,
            profile_picture,
            profile_image,
            profile_image_mimetype
        FROM students
        ORDER BY id DESC
        LIMIT 10
        """,
        fetch=True
    )

    return render_template(
        "dashboard.html",
        total_students=total_students,
        placed_students=placed_students,
        average_cgpa=average_cgpa,
        average_probability=average_probability,
        recent_students=recent_students,
    )


# =========================================================
# STUDENT VALIDATION HELPER
# =========================================================

def get_student_form_data():

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

    attendance = float(
        request.form.get(
            "attendance",
            0
        )
    )

    cgpa = float(
        request.form.get(
            "cgpa",
            0
        )
    )

    internal_marks = float(
        request.form.get(
            "internal_marks",
            0
        )
    )

    projects = int(
        request.form.get(
            "projects",
            0
        )
    )

    skills_score = float(
        request.form.get(
            "skills_score",
            0
        )
    )

    aptitude_score = float(
        request.form.get(
            "aptitude_score",
            0
        )
    )

    communication_score = float(
        request.form.get(
            "communication_score",
            0
        )
    )

    if not roll_no or not name:
        raise ValueError(
            "Roll number and student name are required."
        )

    if department not in [
        "CS",
        "IT",
        "AI",
        "BCA",
        "BBA",
        "B.COM",
    ]:
        raise ValueError(
            "Invalid department."
        )

    if year not in [
        "I",
        "II",
        "III",
    ]:
        raise ValueError(
            "Invalid year."
        )

    if not 0 <= attendance <= 100:
        raise ValueError(
            "Attendance must be between 0 and 100."
        )

    if not 0 <= cgpa <= 10:
        raise ValueError(
            "CGPA must be between 0 and 10."
        )

    if not 0 <= internal_marks <= 100:
        raise ValueError(
            "Internal marks must be between 0 and 100."
        )

    if not 0 <= projects <= 20:
        raise ValueError(
            "Projects must be between 0 and 20."
        )

    if not 0 <= skills_score <= 100:
        raise ValueError(
            "Skills score must be between 0 and 100."
        )

    if not 0 <= aptitude_score <= 100:
        raise ValueError(
            "Aptitude score must be between 0 and 100."
        )

    if not 0 <= communication_score <= 100:
        raise ValueError(
            "Communication score must be between 0 and 100."
        )

    return {
        "roll_no": roll_no,
        "name": name,
        "department": department,
        "year": year,
        "attendance": attendance,
        "cgpa": cgpa,
        "internal_marks": internal_marks,
        "projects": projects,
        "skills_score": skills_score,
        "aptitude_score": aptitude_score,
        "communication_score": communication_score,
    }


# =========================================================
# STUDENT REGISTRATION
# =========================================================

@app.route(
    "/students",
    methods=["GET", "POST"]
)
def students():

    if not session.get("admin"):

        return redirect(
            url_for("login")
        )

    db = DatabaseConnection()

    if request.method == "POST":

        try:

            data = get_student_form_data()

            # -------------------------------------------------
            # CHECK DUPLICATE ROLL NUMBER
            # -------------------------------------------------

            existing_student = db.execute(
                """
                SELECT id
                FROM students
                WHERE roll_no = %s
                """,
                (data["roll_no"],),
                fetchone=True
            )

            if existing_student:

                flash(
                    "Roll number already exists.",
                    "danger"
                )

                return redirect(
                    url_for("students")
                )

            # -------------------------------------------------
            # PROFILE IMAGE
            # -------------------------------------------------

            profile_file = request.files.get(
                "profile_picture"
            )

            profile_image = None
            profile_image_mimetype = None

            if (
                profile_file
                and profile_file.filename
            ):

                if not allowed_image(
                    profile_file.filename
                ):

                    flash(
                        "Only JPG, JPEG, PNG and WEBP images are allowed.",
                        "danger"
                    )

                    return redirect(
                        url_for("students")
                    )

                profile_file.seek(0)

                profile_image = (
                    profile_file.read()
                )

                if not profile_image:

                    flash(
                        "The selected profile image is empty.",
                        "danger"
                    )

                    return redirect(
                        url_for("students")
                    )

                profile_image_mimetype = (
                    get_image_mimetype(
                        profile_file.filename
                    )
                )

            # -------------------------------------------------
            # AI PREDICTION
            # -------------------------------------------------

            (
                performance,
                performance_score,
                placement_probability,
                placement_status,
            ) = predict_student(data)

            # -------------------------------------------------
            # INSERT
            # -------------------------------------------------

            result = db.execute(
                """
                INSERT INTO students (
                    roll_no,
                    name,
                    department,
                    profile_picture,
                    profile_image,
                    profile_image_mimetype,
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
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                RETURNING id
                """,
                (
                    data["roll_no"],
                    data["name"],
                    data["department"],

                    None,

                    # IMPORTANT:
                    # psycopg2.Binary guarantees BYTEA storage.
                    Binary(profile_image)
                    if profile_image
                    else None,

                    profile_image_mimetype,

                    data["year"],
                    data["attendance"],
                    data["cgpa"],
                    data["internal_marks"],
                    data["projects"],
                    data["skills_score"],
                    data["aptitude_score"],
                    data["communication_score"],
                    performance,
                    performance_score,
                    placement_probability,
                    placement_status,
                ),
                fetchone=True
            )

            student_id = result["id"]

            flash(
                "Student registered successfully.",
                "success"
            )

            return redirect(
                url_for(
                    "student_detail",
                    student_id=student_id
                )
            )

        except psycopg2.errors.UniqueViolation:

            flash(
                "Roll number already exists.",
                "danger"
            )

            return redirect(
                url_for("students")
            )

        except ValueError as error:

            flash(
                str(error),
                "danger"
            )

            return redirect(
                url_for("students")
            )

        except Exception as error:

            print(
                "Student registration error:",
                error
            )

            flash(
                f"Unable to register student: {error}",
                "danger"
            )

            return redirect(
                url_for("students")
            )

    # ---------------------------------------------------------
    # GET REGISTERED STUDENTS
    # ---------------------------------------------------------

    registered_students = db.execute(
        """
        SELECT
            id,
            roll_no,
            name,
            department,
            year,
            cgpa,
            placement_probability,
            profile_picture,
            profile_image,
            profile_image_mimetype
        FROM students
        ORDER BY id DESC
        """,
        fetch=True
    )

    return render_template(
        "students.html",
        students=registered_students
    )


# =========================================================
# STUDENT PHOTO
# =========================================================

@app.route(
    "/student-photo/<int:student_id>"
)
def student_photo(student_id):

    if not session.get("admin"):

        return redirect(
            url_for("login")
        )

    db = DatabaseConnection()

    student = db.execute(
        """
        SELECT
            id,
            profile_picture,
            profile_image,
            profile_image_mimetype
        FROM students
        WHERE id = %s
        """,
        (student_id,),
        fetchone=True
    )

    if not student:

        return (
            "Student not found.",
            404
        )

    image_data = student.get(
        "profile_image"
    )

    mimetype = student.get(
        "profile_image_mimetype"
    )

    if image_data:

        return send_file(
            BytesIO(bytes(image_data)),
            mimetype=mimetype or "image/jpeg",
            max_age=0
        )

    # -----------------------------------------------------
    # LEGACY PHOTO RECOVERY
    # -----------------------------------------------------

    old_photo_filename = student.get(
        "profile_picture"
    )

    if old_photo_filename:

        migrated = (
            migrate_legacy_profile_image(
                student_id,
                old_photo_filename
            )
        )

        if migrated:

            recovered_student = db.execute(
                """
                SELECT
                    profile_image,
                    profile_image_mimetype
                FROM students
                WHERE id = %s
                """,
                (student_id,),
                fetchone=True
            )

            if recovered_student:

                recovered_image = (
                    recovered_student.get(
                        "profile_image"
                    )
                )

                recovered_mimetype = (
                    recovered_student.get(
                        "profile_image_mimetype"
                    )
                )

                if recovered_image:

                    return send_file(
                        BytesIO(
                            bytes(recovered_image)
                        ),
                        mimetype=(
                            recovered_mimetype
                            or "image/jpeg"
                        ),
                        max_age=0
                    )

    return (
        "Profile photo not available.",
        404
    )


# =========================================================
# STUDENT DETAIL
# =========================================================

@app.route(
    "/student/<int:student_id>"
)
def student_detail(student_id):

    if not session.get("admin"):

        return redirect(
            url_for("login")
        )

    db = DatabaseConnection()

    student = db.execute(
        """
        SELECT *
        FROM students
        WHERE id = %s
        """,
        (student_id,),
        fetchone=True
    )

    if not student:

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


# =========================================================
# EDIT STUDENT
# =========================================================

@app.route(
    "/student/<int:student_id>/edit",
    methods=["GET", "POST"]
)
def edit_student(student_id):

    if not session.get("admin"):

        return redirect(
            url_for("login")
        )

    db = DatabaseConnection()

    student = db.execute(
        """
        SELECT *
        FROM students
        WHERE id = %s
        """,
        (student_id,),
        fetchone=True
    )

    if not student:

        flash(
            "Student not found.",
            "danger"
        )

        return redirect(
            url_for("students")
        )

    if request.method == "GET":

        return render_template(
            "edit_student.html",
            student=student
        )

    try:

        data = get_student_form_data()

        # -------------------------------------------------
        # DUPLICATE ROLL NUMBER CHECK
        # -------------------------------------------------

        duplicate = db.execute(
            """
            SELECT id
            FROM students
            WHERE roll_no = %s
            AND id != %s
            """,
            (
                data["roll_no"],
                student_id,
            ),
            fetchone=True
        )

        if duplicate:

            flash(
                "Roll number already exists.",
                "danger"
            )

            return redirect(
                url_for(
                    "edit_student",
                    student_id=student_id
                )
            )

        # -------------------------------------------------
        # AI PREDICTION AGAIN
        # -------------------------------------------------

        (
            performance,
            performance_score,
            placement_probability,
            placement_status,
        ) = predict_student(data)

        # -------------------------------------------------
        # CHECK NEW PHOTO
        # -------------------------------------------------

        profile_file = request.files.get(
            "profile_picture"
        )

        has_new_photo = (
            profile_file
            and profile_file.filename
        )

        if has_new_photo:

            if not allowed_image(
                profile_file.filename
            ):

                flash(
                    "Only JPG, JPEG, PNG and WEBP images are allowed.",
                    "danger"
                )

                return redirect(
                    url_for(
                        "edit_student",
                        student_id=student_id
                    )
                )

            profile_file.seek(0)

            profile_image = (
                profile_file.read()
            )

            if not profile_image:

                flash(
                    "The selected profile image is empty.",
                    "danger"
                )

                return redirect(
                    url_for(
                        "edit_student",
                        student_id=student_id
                    )
                )

            profile_image_mimetype = (
                get_image_mimetype(
                    profile_file.filename
                )
            )

            # -------------------------------------------------
            # UPDATE WITH NEW PHOTO
            # -------------------------------------------------

            db.execute(
                """
                UPDATE students
                SET
                    roll_no = %s,
                    name = %s,
                    department = %s,
                    profile_image = %s,
                    profile_image_mimetype = %s,
                    year = %s,
                    attendance = %s,
                    cgpa = %s,
                    internal_marks = %s,
                    projects = %s,
                    skills_score = %s,
                    aptitude_score = %s,
                    communication_score = %s,
                    performance = %s,
                    performance_score = %s,
                    placement_probability = %s,
                    placement_status = %s
                WHERE id = %s
                """,
                (
                    data["roll_no"],
                    data["name"],
                    data["department"],
                    Binary(profile_image),
                    profile_image_mimetype,
                    data["year"],
                    data["attendance"],
                    data["cgpa"],
                    data["internal_marks"],
                    data["projects"],
                    data["skills_score"],
                    data["aptitude_score"],
                    data["communication_score"],
                    performance,
                    performance_score,
                    placement_probability,
                    placement_status,
                    student_id,
                )
            )

        else:

            # -------------------------------------------------
            # UPDATE WITHOUT TOUCHING EXISTING PHOTO
            # -------------------------------------------------

            db.execute(
                """
                UPDATE students
                SET
                    roll_no = %s,
                    name = %s,
                    department = %s,
                    year = %s,
                    attendance = %s,
                    cgpa = %s,
                    internal_marks = %s,
                    projects = %s,
                    skills_score = %s,
                    aptitude_score = %s,
                    communication_score = %s,
                    performance = %s,
                    performance_score = %s,
                    placement_probability = %s,
                    placement_status = %s
                WHERE id = %s
                """,
                (
                    data["roll_no"],
                    data["name"],
                    data["department"],
                    data["year"],
                    data["attendance"],
                    data["cgpa"],
                    data["internal_marks"],
                    data["projects"],
                    data["skills_score"],
                    data["aptitude_score"],
                    data["communication_score"],
                    performance,
                    performance_score,
                    placement_probability,
                    placement_status,
                    student_id,
                )
            )

        flash(
            "Student details updated successfully.",
            "success"
        )

        return redirect(
            url_for(
                "student_detail",
                student_id=student_id
            )
        )

    except psycopg2.errors.UniqueViolation:

        flash(
            "Roll number already exists.",
            "danger"
        )

        return redirect(
            url_for(
                "edit_student",
                student_id=student_id
            )
        )

    except ValueError as error:

        flash(
            str(error),
            "danger"
        )

        return redirect(
            url_for(
                "edit_student",
                student_id=student_id
            )
        )

    except Exception as error:

        print(
            "Student edit error:",
            error
        )

        flash(
            f"Unable to update student: {error}",
            "danger"
        )

        return redirect(
            url_for(
                "edit_student",
                student_id=student_id
            )
        )


# =========================================================
# PHOTO STATUS
# =========================================================

@app.route(
    "/photo-status/<int:student_id>"
)
def photo_status(student_id):

    if not session.get("admin"):

        return redirect(
            url_for("login")
        )

    db = DatabaseConnection()

    student = db.execute(
        """
        SELECT
            id,
            name,
            profile_picture,
            CASE
                WHEN profile_image IS NULL
                THEN 0
                ELSE octet_length(profile_image)
            END AS image_size,
            profile_image_mimetype
        FROM students
        WHERE id = %s
        """,
        (student_id,),
        fetchone=True
    )

    if not student:

        return {
            "error": "Student not found"
        }, 404

    return {
        "id": student["id"],
        "name": student["name"],
        "profile_picture": student["profile_picture"],
        "profile_image_size": student["image_size"],
        "profile_image_mimetype": student["profile_image_mimetype"]
    }


# =========================================================
# DELETE STUDENT
# =========================================================

@app.route(
    "/students/delete/<int:student_id>",
    methods=["POST"]
)
def delete_student(student_id):

    if not session.get("admin"):

        return redirect(
            url_for("login")
        )

    db = DatabaseConnection()

    student = db.execute(
        """
        SELECT id
        FROM students
        WHERE id = %s
        """,
        (student_id,),
        fetchone=True
    )

    if not student:

        flash(
            "Student not found.",
            "danger"
        )

        return redirect(
            url_for("students")
        )

    db.execute(
        """
        DELETE FROM students
        WHERE id = %s
        """,
        (student_id,)
    )

    flash(
        "Student and permanently stored profile photo deleted.",
        "success"
    )

    return redirect(
        url_for("students")
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    try:

        db = DatabaseConnection()

        db.execute(
            "SELECT 1",
            fetchone=True
        )

        return jsonify({
            "status": "ok",
            "database": "connected",
            "profile_storage": "PostgreSQL BYTEA",
            "edit_student": "enabled"
        })

    except Exception as error:

        return jsonify({
            "status": "error",
            "database": str(error)
        }), 500


# =========================================================
# STARTUP
# =========================================================

try:

    initialize_database()

    load_models()

    print(
        "Database initialized successfully."
    )

except Exception as error:

    print(
        "Database initialization error:",
        error
    )


# =========================================================
# LOCAL DEVELOPMENT
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )
