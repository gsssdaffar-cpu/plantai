# -*- coding: utf-8 -*-

"""
============================================================
PlantAI Flask Application
============================================================

Render optimized PlantAI application.

Features:
    - EfficientNet prediction through predict.py
    - 38 plant disease classes
    - AI Highlight ENABLED
    - AI Attention Map DISABLED
    - Lightweight OpenCV highlight
    - Severity estimation
    - Disease information
    - AI explanation
    - Severity advice
    - PDF report
    - Prediction history
    - Dashboard
    - Chatbot
    - Mobile API

IMPORTANT:
    - Only ONE PyTorch model is loaded.
    - No Grad-CAM.
    - No second AI model.
    - Attention map is completely disabled.
    - Highlight uses lightweight OpenCV processing.
    - Severity uses severity.py.
    - Designed for low-memory Render deployment.
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

from disease_lookup import (
    get_disease
)

from history import (
    save_prediction,
    get_history,
    dashboard_stats,
    recent_predictions
)

from severity import (
    estimate_severity
)

from severity_advice import (
    get_severity_advice
)

from report_generator import (
    create_report
)

from database import (
    create_tables
)

from chatbot import (
    chatbot_response
)

from explanation import (
    generate_explanation
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# MAXIMUM UPLOAD SIZE
# ============================================================

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


# ============================================================
# CREATE UPLOAD DIRECTORY
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# ALLOWED IMAGE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

try:

    create_tables()

    print(
        "Database tables initialized successfully."
    )

except Exception as e:

    print(
        "Database initialization error:",
        e
    )


# ============================================================
# MODEL INITIALIZATION
# ============================================================

MODEL = None
TRANSFORM = None

try:

    MODEL = get_model()

    TRANSFORM = get_transform()

    print(
        "PlantAI model initialized successfully."
    )

except Exception as e:

    print(
        "Model initialization error:",
        e
    )

    traceback.print_exc()


# ============================================================
# UTILITY
# ============================================================

def allowed_file(filename):

    if not filename:
        return False

    if "." not in filename:
        return False

    extension = (
        filename
        .rsplit(".", 1)[1]
        .lower()
    )

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# MEMORY CLEANUP
# ============================================================

def safe_cleanup():

    try:

        cleanup()

    except Exception as e:

        print(
            "Prediction cleanup warning:",
            e
        )

    try:

        gc.collect()

    except Exception:

        pass


# ============================================================
# CREATE UNIQUE FILE NAME
# ============================================================

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
# HOME
# ============================================================

@app.route("/")
def home():

    try:

        stats = dashboard_stats()

    except Exception as e:

        print(
            "Dashboard stats error:",
            e
        )

        stats = {}


    try:

        recent = recent_predictions()

    except Exception as e:

        print(
            "Recent predictions error:",
            e
        )

        recent = []


    return render_template(

        "index.html",

        stats=stats,

        recent=recent

    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "ok",

        "service": "PlantAI",

        "model_loaded":
            MODEL is not None,

        "ai_highlight":
            True,

        "ai_attention_map":
            False,

        "severity":
            True,

        "pdf_report":
            True

    })


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

                print(
                    "Chatbot error:",
                    e
                )

                answer = (
                    "Sorry, chatbot is "
                    "currently unavailable."
                )


    return render_template(

        "chatbot.html",

        answer=answer

    )


# ============================================================
# UPLOAD PAGE
# ============================================================

@app.route("/upload")
def upload_page():

    return render_template(
        "upload.html"
    )


# ============================================================
# FALLBACK DISEASE INFORMATION
# ============================================================

def fallback_disease_info(
    prediction
):

    plant = "Unknown"

    disease = prediction

    # Try to recover plant name
    if prediction and " - " in prediction:

        plant = prediction.split(
            " - ",
            1
        )[0].strip()

    return {

        "plant":
            plant,

        "disease":
            disease,

        "cause":
            "Information not available",

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

    severity_level = "Not calculated"

    affected_area = 0.0


    try:

        # ----------------------------------------------------
        # Current severity.py returns:
        #
        # {
        #     "level": "Moderate",
        #     "area": 27.75
        # }
        # ----------------------------------------------------

        result = estimate_severity(
            filepath
        )


        if isinstance(
            result,
            dict
        ):

            # IMPORTANT:
            # Your severity.py uses "level"
            # not "severity".

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

        # Compatibility with a severity.py
        # that expects prediction as second argument.

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


            elif isinstance(
                result,
                tuple
            ):

                if len(result) >= 2:

                    severity_level = result[0]

                    affected_area = result[1]


        except Exception as e:

            print(
                "Severity calculation failed:",
                e
            )


    except Exception as e:

        print(
            "Severity calculation failed:",
            e
        )

        traceback.print_exc()


    # ========================================================
    # CLEAN AREA
    # ========================================================

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


    # ========================================================
    # CLEAN LEVEL
    # ========================================================

    if not severity_level:

        severity_level = (
            "Not calculated"
        )


    severity_level = str(
        severity_level
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

        if result is None:

            return ""

        return str(result)


    except TypeError:

        # Compatibility with older
        # explanation.py

        try:

            result = generate_explanation(

                prediction,

                confidence,

                info

            )

            if result is None:

                return ""

            return str(result)


        except Exception as e:

            print(
                "Explanation error:",
                e
            )

            return ""


    except Exception as e:

        print(
            "Explanation error:",
            e
        )

        return ""


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

        if result is None:

            return {}

        return result


    except TypeError:

        try:

            result = get_severity_advice(
                severity
            )

            if result is None:

                return {}

            return result


        except Exception as e:

            print(
                "Severity advice error:",
                e
            )

            return {}


    except Exception as e:

        print(
            "Severity advice error:",
            e
        )

        return {}


# ============================================================
# AI HIGHLIGHT
# ============================================================

def generate_highlight_safe(
    input_path,
    filename
):

    """
    Lightweight AI-style disease highlight.

    IMPORTANT:

    This does NOT use Grad-CAM.

    It does NOT load another AI model.

    It uses OpenCV to detect visually abnormal
    yellow/brown regions and highlight them.

    This is intentionally lightweight for Render.
    """

    image = None
    hsv = None
    mask = None
    overlay = None

    try:

        print(
            "Generating AI Highlight..."
        )


        # ====================================================
        # READ IMAGE
        # ====================================================

        image = cv2.imread(
            input_path
        )


        if image is None:

            print(
                "AI Highlight: image could not be read."
            )

            return None


        # ====================================================
        # RESIZE
        # ====================================================

        height, width = image.shape[:2]

        max_dimension = 900


        if max(
            height,
            width
        ) > max_dimension:

            scale = (

                max_dimension

                /

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


        # ====================================================
        # HSV
        # ====================================================

        hsv = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2HSV

        )


        # ====================================================
        # DETECT YELLOW / BROWN / DISEASE-LIKE REGIONS
        # ====================================================

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


        # ====================================================
        # MORPHOLOGICAL CLEANUP
        # ====================================================

        kernel = np.ones(

            (
                5,
                5
            ),

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


        # ====================================================
        # REMOVE VERY SMALL REGIONS
        # ====================================================

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
                image.shape[0]
                *
                image.shape[1]
                *
                0.0001
            )

        )


        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area >= minimum_area:

                cv2.drawContours(

                    clean_mask,

                    [contour],

                    -1,

                    255,

                    -1

                )


        mask = clean_mask


        # ====================================================
        # CREATE HIGHLIGHT
        # ====================================================

        overlay = image.copy()


        # Highlight detected regions
        # using red.

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


        # ====================================================
        # APPLY ONLY TO MASK
        # ====================================================

        result = image.copy()


        result[mask > 0] = (
            highlighted[mask > 0]
        )


        # ====================================================
        # DRAW CONTOURS
        # ====================================================

        contours, _ = cv2.findContours(

            mask,

            cv2.RETR_EXTERNAL,

            cv2.CHAIN_APPROX_SIMPLE

        )


        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area >= minimum_area:

                cv2.drawContours(

                    result,

                    [contour],

                    -1,

                    (
                        0,
                        0,
                        255
                    ),

                    2

                )


        # ====================================================
        # OUTPUT NAME
        # ====================================================

        base_name = os.path.splitext(
            filename
        )[0]


        highlight_filename = (

            f"{base_name}_highlight.jpg"

        )


        output_path = os.path.join(

            app.config[
                "UPLOAD_FOLDER"
            ],

            highlight_filename

        )


        # ====================================================
        # SAVE
        # ====================================================

        success = cv2.imwrite(

            output_path,

            result,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                85
            ]

        )


        if not success:

            print(
                "AI Highlight save failed."
            )

            return None


        print(
            "AI Highlight created:",
            highlight_filename
        )


        return highlight_filename


    except Exception as e:

        print(
            "AI Highlight generation failed:",
            e
        )

        traceback.print_exc()

        return None


    finally:

        image = None

        hsv = None

        mask = None

        overlay = None

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

        base_name = os.path.splitext(
            filename
        )[0]


        report_filename = (

            f"{base_name}_report.pdf"

        )


        report_path = os.path.join(

            app.config[
                "UPLOAD_FOLDER"
            ],

            report_filename

        )


        print(
            "Creating PDF report..."
        )


        # ====================================================
        # CURRENT REPORT FORMAT
        # ====================================================

        try:

            result = create_report(

                filename,

                prediction,

                confidence,

                severity,

                affected_area,

                info,

                explanation,

                severity_advice

            )


        except TypeError:

            # =================================================
            # OLD REPORT FORMAT
            # =================================================

            try:

                result = create_report(

                    filename,

                    info,

                    confidence,

                    severity,

                    affected_area

                )


            except TypeError:

                try:

                    result = create_report(

                        filename,

                        info

                    )


                except TypeError:

                    result = create_report(
                        filename
                    )


        # ====================================================
        # CHECK RETURNED PATH
        # ====================================================

        if isinstance(
            result,
            str
        ):

            if os.path.exists(
                result
            ):

                return os.path.basename(
                    result
                )


        if result is not None:

            try:

                result_path = os.fspath(
                    result
                )

                if os.path.exists(
                    result_path
                ):

                    return os.path.basename(
                        result_path
                    )

            except Exception:

                pass


        # ====================================================
        # EXPECTED PATH
        # ====================================================

        if os.path.exists(
            report_path
        ):

            return report_filename


        # ====================================================
        # SEARCH PDF
        # ====================================================

        try:

            for candidate in os.listdir(

                app.config[
                    "UPLOAD_FOLDER"
                ]

            ):

                if (

                    candidate.startswith(
                        base_name
                    )

                    and

                    candidate.lower().endswith(
                        ".pdf"
                    )

                ):

                    return candidate

        except Exception:

            pass


        print(
            "PDF report was not created."
        )

        return None


    except Exception as e:

        print(
            "PDF generation failed:",
            e
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


        print(
            "History saved."
        )

        return True


    except Exception as e:

        print(
            "History save error:",
            e
        )

        return False


# ============================================================
# MAIN WEB UPLOAD
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    print()
    print("=" * 60)
    print("PLANTAI WEB UPLOAD")
    print("=" * 60)


    # ========================================================
    # CHECK FILE
    # ========================================================

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


    # ========================================================
    # VALIDATE
    # ========================================================

    if not allowed_file(
        file.filename
    ):

        return (

            "Invalid image format. "
            "Use JPG, JPEG, PNG or WEBP.",

            400

        )


    # ========================================================
    # UNIQUE FILE
    # ========================================================

    filename = create_unique_filename(
        file.filename
    )


    filepath = os.path.join(

        app.config[
            "UPLOAD_FOLDER"
        ],

        filename

    )


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    try:

        file.save(
            filepath
        )

        print(
            "Image saved:",
            filepath
        )


    except Exception as e:

        print(
            "File save error:",
            e
        )

        return (
            "Could not save image.",
            500
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    try:

        print(
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


        print(
            "Prediction:",
            prediction
        )

        print(
            f"Confidence: {confidence:.2f}%"
        )


    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        traceback.print_exc()

        safe_cleanup()


        return (

            f"Prediction failed: {str(e)}",

            500

        )


    finally:

        safe_cleanup()


    # ========================================================
    # DISEASE INFORMATION
    # ========================================================

    try:

        info = get_disease(
            prediction
        )

    except Exception as e:

        print(
            "Disease lookup error:",
            e
        )

        info = None


    # ========================================================
    # FALLBACK
    # ========================================================

    if info is None:

        print(
            "Disease information not found."
        )

        info = fallback_disease_info(
            prediction
        )


    # ========================================================
    # RECOVER PLANT NAME
    # ========================================================

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


    # ========================================================
    # LOW CONFIDENCE
    # ========================================================

    if confidence < 40:

        return render_template(

            "unknown.html",

            image=filename,

            confidence=round(
                confidence,
                2
            ),

            confidence_status=
                confidence_status,

            top_predictions=
                top_predictions

        )


    # ========================================================
    # SEVERITY
    # ========================================================

    (

        severity_level,

        affected_area

    ) = calculate_severity(

        filepath,

        prediction

    )


    print(
        "Severity:",
        severity_level
    )

    print(
        "Affected area:",
        affected_area
    )


    # ========================================================
    # AI EXPLANATION
    # ========================================================

    explanation = generate_ai_explanation(

        prediction,

        confidence,

        severity_level,

        affected_area,

        info

    )


    # ========================================================
    # SEVERITY ADVICE
    # ========================================================

    severity_advice = (

        generate_severity_advice_safe(

            severity_level,

            prediction,

            affected_area

        )

    )


    # ========================================================
    # AI HIGHLIGHT
    # ========================================================

    print(
        "AI Highlight: ENABLED"
    )


    highlight_name = (

        generate_highlight_safe(

            filepath,

            filename

        )

    )


    # ========================================================
    # AI ATTENTION MAP
    # ========================================================

    # Completely disabled.

    attention_name = None

    print(
        "AI Attention Map: DISABLED"
    )


    # ========================================================
    # HISTORY
    # ========================================================

    save_history_safe(

        filename,

        info,

        prediction,

        confidence,

        severity_level,

        affected_area

    )


    # ========================================================
    # PDF
    # ========================================================

    report_name = (

        generate_report_safe(

            filename,

            prediction,

            confidence,

            severity_level,

            affected_area,

            info,

            explanation,

            severity_advice

        )

    )


    # ========================================================
    # CLEANUP
    # ========================================================

    safe_cleanup()


    # ========================================================
    # LOG RESULT
    # ========================================================

    print()
    print(
        "Rendering result page..."
    )

    print(
        "Original:",
        filename
    )

    print(
        "AI Highlight:",
        highlight_name
    )

    print(
        "AI Attention:",
        attention_name
    )

    print(
        "Severity:",
        severity_level
    )

    print(
        "Affected Area:",
        affected_area
    )

    print(
        "PDF:",
        report_name
    )

    print("=" * 60)


    # ========================================================
    # RESULT PAGE
    # ========================================================

    return render_template(

        "result.html",

        # Original image
        image=filename,

        # AI Highlight
        highlight=highlight_name,

        # Attention disabled
        gradcam=None,

        attention=None,

        # Prediction
        prediction=prediction,

        disease=info.get(
            "disease",
            prediction
        ),

        plant=info.get(
            "plant",
            "Unknown"
        ),

        confidence=round(
            confidence,
            2
        ),

        confidence_status=
            confidence_status,

        top_predictions=
            top_predictions,

        # Severity
        severity=
            severity_level,

        affected_area=
            affected_area,

        # Disease information
        info=
            info,

        # Explanation
        explanation=
            explanation,

        # Advice
        severity_advice=
            severity_advice,

        # PDF
        report=
            report_name

    )


# ============================================================
# MOBILE API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    print()
    print("=" * 60)
    print("PLANTAI MOBILE API")
    print("=" * 60)


    # ========================================================
    # FILE CHECK
    # ========================================================

    if "image" not in request.files:

        return jsonify({

            "success": False,

            "error":
                "No image uploaded"

        }), 400


    file = request.files["image"]


    if file.filename == "":

        return jsonify({

            "success": False,

            "error":
                "No file selected"

        }), 400


    # ========================================================
    # VALIDATE
    # ========================================================

    if not allowed_file(
        file.filename
    ):

        return jsonify({

            "success": False,

            "error":
                "Invalid image format"

        }), 400


    # ========================================================
    # FILE
    # ========================================================

    filename = create_unique_filename(
        file.filename
    )


    filepath = os.path.join(

        app.config[
            "UPLOAD_FOLDER"
        ],

        filename

    )


    try:

        file.save(
            filepath
        )

    except Exception as e:

        return jsonify({

            "success": False,

            "error":
                "Could not save image",

            "details":
                str(e)

        }), 500


    # ========================================================
    # PREDICTION
    # ========================================================

    try:

        (

            prediction,

            confidence,

            top_predictions,

            confidence_status

        ) = predict_image(
            filepath
        )


    except Exception as e:

        print(
            "Prediction error:",
            e
        )

        traceback.print_exc()

        safe_cleanup()


        return jsonify({

            "success": False,

            "error":
                "AI prediction failed",

            "details":
                str(e)

        }), 500


    finally:

        safe_cleanup()


    # ========================================================
    # DISEASE INFO
    # ========================================================

    try:

        info = get_disease(
            prediction
        )

    except Exception as e:

        print(
            "Disease lookup error:",
            e
        )

        info = None


    if info is None:

        info = fallback_disease_info(
            prediction
        )


    # ========================================================
    # PLANT FALLBACK
    # ========================================================

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


    # ========================================================
    # LOW CONFIDENCE
    # ========================================================

    if confidence < 40:

        return jsonify({

            "success":
                True,

            "recognized":
                False,

            "message":
                "Plant could not be identified confidently.",

            "prediction":
                prediction,

            "plant":
                info.get(
                    "plant",
                    "Unknown"
                ),

            "disease":
                info.get(
                    "disease",
                    prediction
                ),

            "confidence":
                round(
                    confidence,
                    2
                ),

            "confidence_status":
                confidence_status,

            "top_predictions":
                top_predictions

        })


    # ========================================================
    # SEVERITY
    # ========================================================

    (

        severity_level,

        affected_area

    ) = calculate_severity(

        filepath,

        prediction

    )


    # ========================================================
    # EXPLANATION
    # ========================================================

    explanation = generate_ai_explanation(

        prediction,

        confidence,

        severity_level,

        affected_area,

        info

    )


    # ========================================================
    # SEVERITY ADVICE
    # ========================================================

    severity_advice = (

        generate_severity_advice_safe(

            severity_level,

            prediction,

            affected_area

        )

    )


    # ========================================================
    # HISTORY
    # ========================================================

    save_history_safe(

        filename,

        info,

        prediction,

        confidence,

        severity_level,

        affected_area

    )


    # ========================================================
    # AI HIGHLIGHT
    # ========================================================

    highlight_name = (

        generate_highlight_safe(

            filepath,

            filename

        )

    )


    # ========================================================
    # ATTENTION DISABLED
    # ========================================================

    attention_name = None


    # ========================================================
    # RESPONSE
    # ========================================================

    response = {

        "success":
            True,

        "recognized":
            True,

        "image":
            filename,

        "prediction":
            prediction,

        "plant":
            info.get(
                "plant",
                "Unknown"
            ),

        "disease":
            info.get(
                "disease",
                prediction
            ),

        "confidence":
            round(
                confidence,
                2
            ),

        "confidence_status":
            confidence_status,

        "severity":
            severity_level,

        "affected_area":
            affected_area,

        "cause":
            info.get(
                "cause",
                ""
            ),

        "symptoms":
            info.get(
                "symptoms",
                []
            ),

        "treatment":
            info.get(
                "treatment",
                []
            ),

        "organic_treatment":
            info.get(
                "organic_treatment",
                []
            ),

        "prevention":
            info.get(
                "prevention",
                []
            ),

        "explanation":
            explanation,

        "severity_advice":
            severity_advice,

        "top_predictions":
            top_predictions,

        # Highlight ENABLED
        "highlight":
            highlight_name,

        # Attention DISABLED
        "attention":
            None

    }


    safe_cleanup()


    return jsonify(
        response
    )


# ============================================================
# HISTORY PAGE
# ============================================================

@app.route("/history")
def history():

    search = request.args.get(
        "search",
        ""
    )


    try:

        rows = get_history(
            search
        )

    except Exception as e:

        print(
            "History error:",
            e
        )

        rows = []


    return render_template(

        "history.html",

        rows=rows,

        search=search

    )


# ============================================================
# SERVE UPLOADED FILES
# ============================================================

@app.route(
    "/uploads/<path:filename>"
)
def uploaded_file(
    filename
):

    return send_from_directory(

        app.config[
            "UPLOAD_FOLDER"
        ],

        filename

    )


# ============================================================
# 413 ERROR
# ============================================================

@app.errorhandler(413)
def file_too_large(error):

    return (

        "Image is too large. "
        "Maximum upload size is 10 MB.",

        413

    )


# ============================================================
# 500 ERROR
# ============================================================

@app.errorhandler(500)
def internal_error(error):

    print(
        "Internal server error:",
        error
    )

    traceback.print_exc()

    safe_cleanup()


    return (

        "PlantAI encountered an internal error. "
        "Please try another image.",

        500

    )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    port = int(

        os.environ.get(
            "PORT",
            5000
        )

    )


    print()
    print("=" * 60)
    print("PLANTAI SERVER")
    print("=" * 60)
    print(
        "Port:",
        port
    )
    print(
        "AI Highlight: ENABLED"
    )
    print(
        "AI Attention Map: DISABLED"
    )
    print(
        "Severity: ENABLED"
    )
    print("=" * 60)


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )