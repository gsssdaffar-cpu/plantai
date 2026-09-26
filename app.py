# -*- coding: utf-8 -*-

"""
============================================================
PlantAI Flask Application & Mobile REST API - V3
============================================================

Production architecture:

    PlantAI V3
        |
        +-- 38-class EfficientNet model
        +-- Disease information
        +-- Severity analysis
        +-- AI highlight
        +-- AI explanation
        +-- Severity advice
        +-- PDF report
        +-- Prediction history
        +-- Dashboard
        +-- Chatbot
        +-- Mobile REST API
        |
        +-- Verified Feedback
                |
                +-- learning/verified/
                |
                +-- Future V3 training
                    (NOT automatic)

IMPORTANT
------------------------------------------------------------
The production model loaded by this application is V3.

Verified feedback DOES NOT modify the production model.

V3 is the current production model.

============================================================
"""

import os
import gc
import uuid
import traceback

import cv2
import numpy as np

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    jsonify
)

from werkzeug.utils import secure_filename


# ============================================================
# PROJECT IMPORTS
# ============================================================

from predict import (
    predict_image,
    get_model,
    get_transform,
    cleanup
)

from disease_lookup import get_disease

from history import (
    save_prediction,
    get_history,
    dashboard_stats,
    recent_predictions
)

from severity import estimate_severity

from severity_advice import get_severity_advice

from report_generator import create_report

from database import create_tables

from chatbot import chatbot_response

from explanation import generate_explanation


# ============================================================
# VERIFIED FEEDBACK
# ============================================================

from feedback import (
    save_verified_feedback,
    get_learning_status,
    get_model_classes
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Maximum upload size = 10 MB
app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)

# Supported image formats
ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# PRODUCTION MODEL IDENTITY
# ============================================================

# IMPORTANT:
# This application is running the V3 production model.
#
# Verified feedback may contain future training samples, but the V3
# production model is loaded directly by this application.

PRODUCTION_MODEL = "V3"

MODEL_VERSION = "PlantAI V3"

NUM_CLASSES = 38


# ============================================================
# LOGGING
# ============================================================

def log(message):

    print(
        message,
        flush=True
    )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

try:

    log(
        "STEP 0: Initializing database..."
    )

    create_tables()

    log(
        "STEP 0: Database initialized successfully."
    )

except Exception as e:

    log(
        f"STEP 0 ERROR: Database initialization failed: {e}"
    )

    traceback.print_exc()


# ============================================================
# MODEL INITIALIZATION
# ============================================================

MODEL = None
TRANSFORM = None

try:

    log(
        "STEP 0: Loading PlantAI V3 model..."
    )

    MODEL = get_model()

    TRANSFORM = get_transform()

    log(
        "STEP 0: PlantAI V3 model loaded successfully."
    )

except Exception as e:

    log(
        f"STEP 0 ERROR: Model initialization failed: {e}"
    )

    traceback.print_exc()


# ============================================================
# FEEDBACK SYSTEM INITIALIZATION
# ============================================================

feedback_status = {
    "verified_samples": 0,
    "production_model": PRODUCTION_MODEL
}

try:

    log(
        "STEP 0: Initializing verified feedback system..."
    )

    feedback_status = get_learning_status()

    log(
        "STEP 0: Feedback system initialized."
    )

    log(
        "STEP 0: Verified samples = "
        f"{feedback_status.get('verified_samples', 0)}"
    )

    # DO NOT trust feedback.py to determine the currently
    # loaded production model.
    #
    # The current application explicitly uses V3.

    log(
        "STEP 0: Production model = "
        f"{PRODUCTION_MODEL}"
    )

except Exception as e:

    log(
        f"STEP 0 ERROR: Feedback initialization failed: {e}"
    )

    traceback.print_exc()


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def allowed_file(filename):

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


def safe_cleanup():

    try:

        cleanup()

    except Exception as e:

        log(
            f"Cleanup warning: {e}"
        )

    try:

        gc.collect()

    except Exception:

        pass


def create_unique_filename(
    original_filename
):

    safe_name = secure_filename(
        original_filename
    )

    if not safe_name:

        safe_name = "plant.jpg"

    unique_id = uuid.uuid4().hex[:12]

    return (
        f"{unique_id}_{safe_name}"
    )


# ============================================================
# FALLBACK DISEASE INFORMATION
# ============================================================

def fallback_disease_info(
    prediction
):

    plant = "Unknown"

    disease = prediction

    if (
        prediction
        and
        " - " in prediction
    ):

        plant = prediction.split(
            " - ",
            1
        )[0].strip()

    return {

        "plant": plant,

        "disease": disease,

        "cause": "Information not available",

        "symptoms": [
            "No disease information available"
        ],

        "treatment": [
            "Consult appropriate plant disease guidance"
        ],

        "organic_treatment": [
            "Maintain normal plant care"
        ],

        "prevention": [
            "Monitor the plant regularly"
        ]

    }


# ============================================================
# SEVERITY
# ============================================================

def calculate_severity(
    filepath,
    prediction=None
):

    # Healthy plants do not require disease severity analysis.

    if (
        prediction
        and
        "healthy" in prediction.lower()
    ):

        log(
            "SEVERITY: Healthy prediction detected; "
            "skipping severity analysis."
        )

        return (
            "Healthy",
            0.0
        )

    severity_level = (
        "Not calculated"
    )

    affected_area = 0.0

    try:

        log(
            "SEVERITY: Starting severity analysis..."
        )

        result = estimate_severity(
            filepath
        )

        if isinstance(
            result,
            dict
        ):

            severity_level = result.get(
                "level",
                result.get(
                    "severity",
                    result.get(
                        "severity_level",
                        "Not calculated"
                    )
                )
            )

            affected_area = result.get(
                "area",
                result.get(
                    "affected_area",
                    result.get(
                        "affected",
                        0
                    )
                )
            )

        elif isinstance(
            result,
            tuple
        ):

            if len(result) >= 2:

                severity_level = result[0]

                affected_area = result[1]

            elif len(result) == 1:

                severity_level = result[0]

        elif isinstance(
            result,
            str
        ):

            severity_level = result

        elif isinstance(
            result,
            (int, float)
        ):

            affected_area = float(
                result
            )

    except TypeError:

        # Compatibility with severity functions that expect
        # prediction as a second parameter.

        try:

            result = estimate_severity(
                filepath,
                prediction
            )

            if isinstance(
                result,
                dict
            ):

                severity_level = result.get(
                    "level",
                    result.get(
                        "severity",
                        "Not calculated"
                    )
                )

                affected_area = result.get(
                    "area",
                    result.get(
                        "affected_area",
                        0
                    )
                )

            elif (
                isinstance(
                    result,
                    tuple
                )
                and
                len(result) >= 2
            ):

                severity_level = result[0]

                affected_area = result[1]

        except Exception as e:

            log(
                f"SEVERITY fallback failed: {e}"
            )

    except Exception as e:

        log(
            f"SEVERITY calculation failed: {e}"
        )

        traceback.print_exc()

    try:

        affected_area = float(
            affected_area
        )

    except Exception:

        affected_area = 0.0

    affected_area = max(
        0.0,
        min(
            100.0,
            affected_area
        )
    )

    severity_level = (
        str(severity_level)
        if severity_level
        else
        "Not calculated"
    )

    log(
        f"SEVERITY: Completed -> "
        f"{severity_level}, "
        f"{affected_area:.2f}%"
    )

    return (
        severity_level,
        round(
            affected_area,
            2
        )
    )


# ============================================================
# AI EXPLANATION
# ============================================================

def generate_ai_explanation(
    prediction,
    confidence,
    severity,
    affected_area,
    info
):

    try:

        result = generate_explanation(
            prediction,
            confidence,
            severity,
            affected_area,
            info
        )

        if result is not None:

            return str(
                result
            )

    except Exception:

        pass

    try:

        result = generate_explanation(
            prediction,
            confidence,
            severity,
            affected_area
        )

        if result is not None:

            return str(
                result
            )

    except Exception:

        pass

    try:

        result = generate_explanation(
            prediction,
            confidence,
            info
        )

        if result is not None:

            return str(
                result
            )

    except Exception:

        pass

    return (
        "Explanation currently unavailable."
    )


# ============================================================
# SEVERITY ADVICE
# ============================================================

def generate_severity_advice_safe(
    severity,
    prediction,
    affected_area
):

    try:

        result = get_severity_advice(
            severity,
            affected_area,
            prediction
        )

        return (
            result
            if result is not None
            else {}
        )

    except TypeError:

        try:

            result = get_severity_advice(
                severity
            )

            return (
                result
                if result is not None
                else {}
            )

        except Exception as e:

            log(
                f"Severity advice fallback error: {e}"
            )

            return {}

    except Exception as e:

        log(
            f"Severity advice error: {e}"
        )

        return {}


# ============================================================
# AI HIGHLIGHT
# ============================================================

def generate_highlight_safe(
    input_path,
    filename
):

    try:

        log(
            "HIGHLIGHT: Starting..."
        )

        image = cv2.imread(
            input_path
        )

        if image is None:

            log(
                "HIGHLIGHT: Image could not be read."
            )

            return None

        height, width = image.shape[:2]

        max_dimension = 900

        if max(
            height,
            width
        ) > max_dimension:

            scale = (
                max_dimension /
                float(
                    max(
                        height,
                        width
                    )
                )
            )

            new_width = max(
                1,
                int(
                    width * scale
                )
            )

            new_height = max(
                1,
                int(
                    height * scale
                )
            )

            image = cv2.resize(
                image,
                (
                    new_width,
                    new_height
                ),
                interpolation=cv2.INTER_AREA
            )

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        lower_disease = np.array(
            [
                8,
                35,
                30
            ],
            dtype=np.uint8
        )

        upper_disease = np.array(
            [
                45,
                255,
                255
            ],
            dtype=np.uint8
        )

        mask = cv2.inRange(
            hsv,
            lower_disease,
            upper_disease
        )

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        clean_mask = np.zeros_like(
            mask
        )

        minimum_area = max(
            20,
            int(
                image.shape[0] *
                image.shape[1] *
                0.0001
            )
        )

        for contour in contours:

            if (
                cv2.contourArea(
                    contour
                )
                >= minimum_area
            ):

                cv2.drawContours(
                    clean_mask,
                    [contour],
                    -1,
                    255,
                    -1
                )

        mask = clean_mask

        overlay = image.copy()

        red_layer = np.zeros_like(
            image
        )

        red_layer[:, :, 2] = 255

        highlighted = cv2.addWeighted(
            overlay,
            0.65,
            red_layer,
            0.35,
            0
        )

        result = image.copy()

        result[
            mask > 0
        ] = highlighted[
            mask > 0
        ]

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            if (
                cv2.contourArea(
                    contour
                )
                >= minimum_area
            ):

                cv2.drawContours(
                    result,
                    [contour],
                    -1,
                    (0, 0, 255),
                    2
                )

        base_name = os.path.splitext(
            filename
        )[0]

        highlight_filename = (
            f"{base_name}_highlight.jpg"
        )

        output_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            highlight_filename
        )

        success = cv2.imwrite(
            output_path,
            result,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                85
            ]
        )

        if not success:

            log(
                "HIGHLIGHT: Save failed."
            )

            return None

        log(
            "HIGHLIGHT: Completed."
        )

        return highlight_filename

    except Exception as e:

        log(
            f"HIGHLIGHT ERROR: {e}"
        )

        traceback.print_exc()

        return None

    finally:

        gc.collect()


# ============================================================
# PDF REPORT
# ============================================================

def generate_report_safe(
    filename,
    prediction,
    confidence,
    severity,
    affected_area,
    info,
    explanation,
    severity_advice
):

    try:

        log(
            "REPORT: Starting PDF generation..."
        )

        base_name = os.path.splitext(
            filename
        )[0]

        report_filename = (
            f"{base_name}_report.pdf"
        )

        plant = info.get(
            "plant",
            "Unknown"
        )

        disease = info.get(
            "disease",
            prediction
        )

        cause = info.get(
            "cause",
            "N/A"
        )

        symptoms = info.get(
            "symptoms",
            []
        )

        treatment = info.get(
            "treatment",
            []
        )

        organic_treatment = info.get(
            "organic_treatment",
            []
        )

        prevention = info.get(
            "prevention",
            []
        )

        result = create_report(
            filename,
            plant,
            disease,
            confidence,
            severity,
            affected_area,
            cause,
            symptoms,
            treatment,
            organic_treatment,
            prevention
        )

        if (
            result
            and
            os.path.exists(result)
        ):

            log(
                "REPORT: Completed."
            )

            return os.path.basename(
                result
            )

        target_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            report_filename
        )

        if os.path.exists(
            target_path
        ):

            log(
                "REPORT: Completed."
            )

            return report_filename

        log(
            "REPORT: File was not created."
        )

        return None

    except Exception as e:

        log(
            f"REPORT ERROR: {e}"
        )

        traceback.print_exc()

        return None


# ============================================================
# HISTORY
# ============================================================

def save_history_safe(
    filename,
    info,
    prediction,
    confidence,
    severity,
    affected_area
):

    try:

        save_prediction(
            filename,
            info.get(
                "plant",
                "Unknown"
            ),
            info.get(
                "disease",
                prediction
            ),
            confidence,
            severity,
            affected_area
        )

        log(
            "HISTORY: Saved."
        )

        return True

    except Exception as e:

        log(
            f"HISTORY ERROR: {e}"
        )

        return False


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    log(
        "HEALTH CHECK received."
    )

    try:

        learning_status = (
            get_learning_status()
        )

    except Exception:

        learning_status = {
            "verified_samples": 0
        }

    return jsonify({

        "status": "ok",

        "service": "PlantAI API",

        "model_loaded": (
            MODEL is not None
        ),

        "model": MODEL_VERSION,

        "classes": NUM_CLASSES,

        "ai_highlight": True,

        "severity": True,

        "pdf_report": True,

        "feedback_enabled": True,

        "verified_samples": (
            learning_status.get(
                "verified_samples",
                0
            )
        ),

        # IMPORTANT:
        # Always report the actual production model
        # currently configured by this application.

        "production_model": (
            PRODUCTION_MODEL
        )

    })


# ============================================================
# WEB HOME
# ============================================================

@app.route(
    "/"
)
def home():

    try:

        stats = dashboard_stats()

    except Exception as e:

        log(
            f"Dashboard stats error: {e}"
        )

        stats = {}

    try:

        recent = recent_predictions()

    except Exception as e:

        log(
            f"Recent predictions error: {e}"
        )

        recent = []

    return render_template(
        "index.html",
        stats=stats,
        recent=recent
    )


# ============================================================
# CHATBOT
# ============================================================

@app.route(
    "/chatbot",
    methods=["GET", "POST"]
)
def chatbot():

    answer = ""

    if request.method == "POST":

        question = request.form.get(
            "question",
            ""
        ).strip()

        if question:

            try:

                answer = chatbot_response(
                    question
                )

            except Exception as e:

                log(
                    f"Chatbot error: {e}"
                )

                answer = (
                    "Sorry, chatbot is currently unavailable."
                )

    return render_template(
        "chatbot.html",
        answer=answer
    )


# ============================================================
# UPLOADED FILES
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ============================================================
# UPLOAD PAGE
# ============================================================

@app.route(
    "/upload",
    methods=["GET"]
)
def upload_page():

    return render_template(
        "upload.html"
    )


# ============================================================
# WEB UPLOAD
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    log("")
    log("=" * 60)
    log("PLANTAI WEB UPLOAD")
    log("=" * 60)

    if "image" not in request.files:

        return (
            "No image uploaded.",
            400
        )

    file = request.files["image"]

    if file.filename == "":

        return (
            "No file selected.",
            400
        )

    if not allowed_file(
        file.filename
    ):

        return (
            "Invalid image format. "
            "Use JPG, JPEG, PNG or WEBP.",
            400
        )

    filename = create_unique_filename(
        file.filename
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    try:

        file.save(
            filepath
        )

        log(
            f"Image saved: {filepath}"
        )

    except Exception as e:

        log(
            f"File save error: {e}"
        )

        return (
            "Could not save image.",
            500
        )

    try:

        log(
            "Starting AI prediction..."
        )

        (
            prediction,
            confidence,
            top_predictions,
            confidence_status
        ) = predict_image(
            filepath
        )

        log(
            f"Prediction: {prediction}"
        )

        log(
            f"Confidence: {confidence:.2f}%"
        )

    except Exception as e:

        log(
            f"Prediction error: {e}"
        )

        traceback.print_exc()

        return (
            f"Prediction failed: {str(e)}",
            500
        )

    finally:

        safe_cleanup()

    try:

        info = get_disease(
            prediction
        )

    except Exception as e:

        log(
            f"Disease lookup error: {e}"
        )

        info = None

    if info is None:

        info = fallback_disease_info(
            prediction
        )

    if (
        info.get(
            "plant",
            "Unknown"
        ) == "Unknown"
        and
        prediction
        and
        " - " in prediction
    ):

        info["plant"] = prediction.split(
            " - ",
            1
        )[0].strip()

    # Very low confidence prediction
    if confidence < 35.0:

        return render_template(
            "unknown.html",
            image=filename,
            confidence=round(
                confidence,
                2
            ),
            confidence_status=confidence_status,
            top_predictions=top_predictions
        )

    severity_level, affected_area = (
        calculate_severity(
            filepath,
            prediction
        )
    )

    explanation = generate_ai_explanation(
        prediction,
        confidence,
        severity_level,
        affected_area,
        info
    )

    severity_advice = (
        generate_severity_advice_safe(
            severity_level,
            prediction,
            affected_area
        )
    )

    if (
        "healthy"
        in prediction.lower()
    ):

        highlight_file = None

    else:

        highlight_file = (
            generate_highlight_safe(
                filepath,
                filename
            )
        )

    report_file = generate_report_safe(
        filename,
        prediction,
        confidence,
        severity_level,
        affected_area,
        info,
        explanation,
        severity_advice
    )

    save_history_safe(
        filename,
        info,
        prediction,
        confidence,
        severity_level,
        affected_area
    )

    return render_template(
        "result.html",
        image=filename,
        highlight_image=highlight_file,
        report_file=report_file,
        prediction=prediction,
        confidence=round(
            confidence,
            2
        ),
        confidence_status=confidence_status,
        top_predictions=top_predictions,
        severity=severity_level,
        affected_area=affected_area,
        info=info,
        explanation=explanation,
        severity_advice=severity_advice
    )


# ============================================================
# MOBILE REST API - V2 PREDICTION
# ============================================================

@app.route(
    "/api/v1/predict",
    methods=["POST"]
)
def api_predict():

    filepath = None
    filename = None

    log("")
    log("=" * 60)
    log("PLANTAI V2 MOBILE API REQUEST")
    log("=" * 60)

    try:

        # ====================================================
        # STEP 1 - VALIDATE REQUEST
        # ====================================================

        log(
            "STEP 1: Validating request..."
        )

        if "image" not in request.files:

            log(
                "STEP 1 ERROR: No image field."
            )

            return jsonify({

                "success": False,

                "error": (
                    "No image field found in request."
                )

            }), 400

        file = request.files["image"]

        if file.filename == "":

            log(
                "STEP 1 ERROR: Empty filename."
            )

            return jsonify({

                "success": False,

                "error": (
                    "No file selected."
                )

            }), 400

        if not allowed_file(
            file.filename
        ):

            log(
                "STEP 1 ERROR: Unsupported format."
            )

            return jsonify({

                "success": False,

                "error": (
                    "Unsupported image format. "
                    "Use JPG, JPEG, PNG or WEBP."
                )

            }), 400

        log(
            "STEP 1: Request validated."
        )

        # ====================================================
        # STEP 2 - SAVE IMAGE
        # ====================================================

        log(
            "STEP 2: Saving uploaded image..."
        )

        filename = create_unique_filename(
            file.filename
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(
            filepath
        )

        log(
            f"STEP 2: Image saved: {filename}"
        )

        # ====================================================
        # STEP 3 - AI PREDICTION
        # ====================================================

        log(
            "STEP 3: Starting predict_image()..."
        )

        log(
            "STEP 3: If Render stops here, "
            "model inference is the bottleneck."
        )

        (
            prediction,
            confidence,
            top_predictions,
            confidence_status
        ) = predict_image(
            filepath
        )

        log(
            "STEP 3: predict_image() COMPLETED."
        )

        log(
            f"STEP 3: Prediction = {prediction}"
        )

        log(
            f"STEP 3: Confidence = {confidence:.2f}%"
        )

        # ====================================================
        # STEP 4 - DISEASE LOOKUP
        # ====================================================

        log(
            "STEP 4: Starting disease lookup..."
        )

        try:

            info = get_disease(
                prediction
            )

        except Exception as e:

            log(
                f"STEP 4 ERROR: {e}"
            )

            info = None

        if info is None:

            info = fallback_disease_info(
                prediction
            )

        log(
            "STEP 4: Disease lookup completed."
        )

        # ====================================================
        # STEP 5 - NORMALIZE INFORMATION
        # ====================================================

        log(
            "STEP 5: Normalizing disease information..."
        )

        plant_name = info.get(
            "plant",
            "Unknown Plant"
        )

        disease_name = info.get(
            "disease",
            prediction
        )

        cause = info.get(
            "cause",
            "N/A"
        )

        symptoms = info.get(
            "symptoms",
            []
        )

        treatment = info.get(
            "treatment",
            []
        )

        organic_treatment = info.get(
            "organic_treatment",
            []
        )

        prevention = info.get(
            "prevention",
            []
        )

        log(
            "STEP 5: Information normalized."
        )

        # ====================================================
        # STEP 6 - SEVERITY
        # ====================================================

        log(
            "STEP 6: Starting severity analysis..."
        )

        try:

            (
                severity_level,
                affected_area
            ) = calculate_severity(
                filepath,
                prediction
            )

        except Exception as e:

            log(
                f"STEP 6 ERROR: {e}"
            )

            severity_level = "Unknown"

            affected_area = 0.0

        log(
            f"STEP 6: Severity = {severity_level}"
        )

        log(
            "STEP 6: Affected area = "
            f"{affected_area:.2f}%"
        )

        # ====================================================
        # STEP 7 - EXPLANATION
        # ====================================================

        log(
            "STEP 7: Starting explanation..."
        )

        try:

            explanation = generate_ai_explanation(
                prediction,
                confidence,
                severity_level,
                affected_area,
                info
            )

        except Exception as e:

            log(
                f"STEP 7 ERROR: {e}"
            )

            explanation = ""

        log(
            "STEP 7: Explanation completed."
        )

        # ====================================================
        # STEP 8 - SEVERITY ADVICE
        # ====================================================

        log(
            "STEP 8: Starting severity advice..."
        )

        try:

            severity_advice = (
                generate_severity_advice_safe(
                    severity_level,
                    prediction,
                    affected_area
                )
            )

        except Exception as e:

            log(
                f"STEP 8 ERROR: {e}"
            )

            severity_advice = {}

        log(
            "STEP 8: Severity advice completed."
        )

        # ====================================================
        # STEP 9 - PREDICTION MARGIN
        # ====================================================

        log(
            "STEP 9: Calculating prediction margin..."
        )

        prediction_margin = 100.0

        try:

            if (
                isinstance(
                    top_predictions,
                    list
                )
                and
                len(
                    top_predictions
                ) >= 2
            ):

                first = float(
                    top_predictions[0][
                        "confidence"
                    ]
                )

                second = float(
                    top_predictions[1][
                        "confidence"
                    ]
                )

                prediction_margin = (
                    first - second
                )

        except Exception as e:

            log(
                f"STEP 9 ERROR: {e}"
            )

        log(
            "STEP 9: Prediction margin = "
            f"{prediction_margin:.2f}%"
        )

        # ====================================================
        # STEP 10 - UNCERTAINTY
        # ====================================================

        log(
            "STEP 10: Calculating uncertainty..."
        )

        low_confidence = (
            confidence < 70.0
        )

        ambiguous_prediction = (
            prediction_margin < 10.0
        )

        uncertain = (
            low_confidence
            or
            ambiguous_prediction
        )

        log(
            f"STEP 10: Uncertain = {uncertain}"
        )

        # ====================================================
        # STEP 11 - BUILD RESPONSE
        # ====================================================

        log(
            "STEP 11: Building JSON response..."
        )

        response = {

            "success": True,

            "model": MODEL_VERSION,

            "model_version": MODEL_VERSION,

            "production_model": (
                PRODUCTION_MODEL
            ),

            "classes": NUM_CLASSES,

            "plant": str(
                plant_name
            ),

            "disease": str(
                disease_name
            ),

            "prediction": str(
                prediction
            ),

            "confidence": (
                f"{confidence:.2f}%"
            ),

            "confidence_value": round(
                float(confidence),
                2
            ),

            "confidence_status": (
                confidence_status
            ),

            "prediction_margin": (
                f"{prediction_margin:.2f}%"
            ),

            "prediction_margin_value": round(
                float(prediction_margin),
                2
            ),

            "uncertain": uncertain,

            "low_confidence": (
                low_confidence
            ),

            "ambiguous_prediction": (
                ambiguous_prediction
            ),

            "top_predictions": (
                top_predictions
            ),

            "severity": str(
                severity_level
            ),

            "leaf_area_affected": (
                f"{affected_area:.2f}%"
            ),

            "affected_area_value": round(
                float(affected_area),
                2
            ),

            "cause": str(
                cause
            ),

            "symptoms": symptoms,

            "treatment": treatment,

            "organic_treatment": (
                organic_treatment
            ),

            "prevention": prevention,

            "explanation": (
                explanation
            ),

            "severity_advice": (
                severity_advice
            ),

            "ai_highlight": True,

            "severity_analysis": True,

            "pdf_report": True,

            "feedback_enabled": True,

            "feedback_endpoint": (
                "/api/v1/feedback"
            )

        }

        # ====================================================
        # STEP 12 - RESPONSE READY
        # ====================================================

        log(
            "STEP 12: Mobile API response ready."
        )

        log(
            "=" * 60
        )

        return jsonify(
            response
        ), 200

    except Exception as e:

        log("")
        log("=" * 60)
        log("PLANTAI V2 MOBILE API ERROR")
        log("=" * 60)

        log(
            f"ERROR TYPE: {type(e).__name__}"
        )

        log(
            f"ERROR: {str(e)}"
        )

        traceback.print_exc()

        log(
            "=" * 60
        )

        return jsonify({

            "success": False,

            "model": MODEL_VERSION,

            "error": str(e)

        }), 500

    finally:

        if filepath:

            try:

                if os.path.exists(
                    filepath
                ):

                    os.remove(
                        filepath
                    )

                    log(
                        "CLEANUP: Removed temporary image "
                        f"{filename}"
                    )

            except Exception as e:

                log(
                    f"CLEANUP ERROR: {e}"
                )


# ============================================================
# MOBILE REST API - VERIFIED USER FEEDBACK
# ============================================================

@app.route(
    "/api/v1/feedback",
    methods=["POST"]
)
def api_feedback():

    filepath = None
    filename = None

    log("")
    log("=" * 60)
    log("PLANTAI V2 VERIFIED FEEDBACK REQUEST")
    log("=" * 60)

    try:

        # ====================================================
        # STEP 1 - VALIDATE REQUEST
        # ====================================================

        log(
            "FEEDBACK STEP 1: Validating request..."
        )

        if "image" not in request.files:

            return jsonify({

                "success": False,

                "error": (
                    "No image field found in feedback request."
                )

            }), 400

        file = request.files["image"]

        if file.filename == "":

            return jsonify({

                "success": False,

                "error": "No feedback image selected."

            }), 400

        if not allowed_file(
            file.filename
        ):

            return jsonify({

                "success": False,

                "error": (
                    "Unsupported image format. "
                    "Use JPG, JPEG, PNG or WEBP."
                )

            }), 400

        # ====================================================
        # STEP 2 - READ FEEDBACK DATA
        # ====================================================

        predicted_class = request.form.get(
            "predicted_class",
            ""
        ).strip()

        correct_class = request.form.get(
            "correct_class",
            ""
        ).strip()

        confidence_raw = request.form.get(
            "confidence",
            "0"
        ).strip()

        feedback_type = request.form.get(
            "feedback_type",
            "user_verified"
        ).strip()

        if not correct_class:

            return jsonify({

                "success": False,

                "error": (
                    "correct_class is required."
                )

            }), 400

        # ====================================================
        # STEP 3 - VALIDATE CONFIDENCE
        # ====================================================

        try:

            confidence = float(
                confidence_raw
            )

        except Exception:

            confidence = 0.0

        confidence = max(
            0.0,
            min(
                100.0,
                confidence
            )
        )

        # ====================================================
        # STEP 4 - VALIDATE V2 CLASS
        # ====================================================

        log(
            "FEEDBACK STEP 4: "
            "Validating PlantAI V2 class..."
        )

        classes = get_model_classes()

        matched_class = None

        for known_class in classes:

            if known_class == correct_class:

                matched_class = known_class

                break

        if matched_class is None:

            for known_class in classes:

                if known_class.lower() == (
                    correct_class.lower()
                ):

                    matched_class = known_class

                    break

        if matched_class is None:

            return jsonify({

                "success": False,

                "error": (
                    "Invalid correct_class. "
                    "The supplied class does not "
                    "exist in the PlantAI V2 38-class model."
                )

            }), 400

        correct_class = matched_class

        # ====================================================
        # STEP 5 - SAVE TEMPORARY IMAGE
        # ====================================================

        log(
            "FEEDBACK STEP 5: "
            "Saving feedback image..."
        )

        filename = create_unique_filename(
            file.filename
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(
            filepath
        )

        log(
            "FEEDBACK STEP 5: "
            f"Image saved: {filename}"
        )

        # ====================================================
        # STEP 6 - STORE VERIFIED SAMPLE
        # ====================================================

        log(
            "FEEDBACK STEP 6: "
            "Saving verified sample..."
        )

        result = save_verified_feedback(

            image_path=filepath,

            correct_class=correct_class,

            predicted_class=predicted_class,

            confidence=confidence,

            feedback_type=feedback_type

        )

        log(
            "FEEDBACK STEP 6: "
            "Verified sample saved."
        )

        # ====================================================
        # STEP 7 - UPDATED LEARNING STATUS
        # ====================================================

        log(
            "FEEDBACK STEP 7: "
            "Reading learning status..."
        )

        status = get_learning_status()

        verified_samples = status.get(
            "verified_samples",
            0
        )

        log(
            "FEEDBACK STEP 7: "
            f"Verified samples = {verified_samples}"
        )

        # ====================================================
        # STEP 8 - RESPONSE
        # ====================================================

        response = {

            "success": True,

            "message": (
                "Feedback verified and stored successfully."
            ),

            "model": MODEL_VERSION,

            "production_model": (
                PRODUCTION_MODEL
            ),

            "predicted_class": (
                predicted_class
            ),

            "correct_class": (
                correct_class
            ),

            "confidence": (
                f"{confidence:.2f}%"
            ),

            "confidence_value": round(
                confidence,
                2
            ),

            "feedback_type": (
                feedback_type
            ),

            "saved_filename": (
                result.get(
                    "filename"
                )
            ),

            "verified_samples": (
                verified_samples
            ),

            # Critical safety flag:
            # feedback never modifies the V2 model.

            "model_modified": False,

            "v3_training_required": True

        }

        log(
            "FEEDBACK STEP 8: "
            "Response ready."
        )

        log(
            "=" * 60
        )

        return jsonify(
            response
        ), 200

    except Exception as e:

        log(
            "PLANTAI FEEDBACK ERROR"
        )

        log(
            f"ERROR TYPE: {type(e).__name__}"
        )

        log(
            f"ERROR: {str(e)}"
        )

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500

    finally:

        # The feedback engine should have copied the image
        # into learning/verified/<class_name>/.
        #
        # The temporary upload is therefore deleted.

        if filepath:

            try:

                if os.path.exists(
                    filepath
                ):

                    os.remove(
                        filepath
                    )

                    log(
                        "FEEDBACK CLEANUP: "
                        f"Removed temporary image {filename}"
                    )

            except Exception as e:

                log(
                    f"FEEDBACK CLEANUP ERROR: {e}"
                )


# ============================================================
# LEARNING STATUS API
# ============================================================

@app.route(
    "/api/v1/learning/status",
    methods=["GET"]
)
def api_learning_status():

    try:

        status = get_learning_status()

        # Never allow the feedback subsystem's metadata
        # to incorrectly identify the currently loaded model.

        status["production_model"] = (
            PRODUCTION_MODEL
        )

        status["model_version"] = (
            MODEL_VERSION
        )

        status["classes"] = (
            NUM_CLASSES
        )

        status["model_modified"] = False

        return jsonify({

            "success": True,

            "learning": status

        }), 200

    except Exception as e:

        log(
            f"LEARNING STATUS ERROR: {e}"
        )

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# SERVER START
# ============================================================

if __name__ == "__main__":

    log("")
    log("=" * 60)
    log("PLANTAI PRODUCTION SERVER")
    log("=" * 60)

    log(
        f"BASE DIR: {BASE_DIR}"
    )

    log(
        f"UPLOAD FOLDER: {UPLOAD_FOLDER}"
    )

    log(
        f"MODEL LOADED: {MODEL is not None}"
    )

    log(
        f"MODEL VERSION: {MODEL_VERSION}"
    )

    log(
        f"PRODUCTION MODEL: {PRODUCTION_MODEL}"
    )

    log(
        f"CLASSES: {NUM_CLASSES}"
    )

    try:

        status = get_learning_status()

        log(
            "VERIFIED SAMPLES: "
            f"{status.get('verified_samples', 0)}"
        )

    except Exception as e:

        log(
            f"LEARNING STATUS ERROR: {e}"
        )

    log("=" * 60)

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )