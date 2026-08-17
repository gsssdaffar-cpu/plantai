# -*- coding: utf-8 -*-

"""
============================================================
PlantAI Flask Application
============================================================

Render 512 MB memory optimized version.

Features:
    - Plant disease prediction
    - Disease information
    - Severity estimation
    - AI explanation
    - AI Highlight DISABLED
    - Lightweight AI Attention Map
    - PDF report
    - Prediction history
    - Dashboard
    - Chatbot
    - Mobile API

IMPORTANT:
    - predict.py contains the main EfficientNet model
    - Only ONE prediction model is loaded
    - No pytorch-grad-cam
    - No second AI model
    - AI Highlight is disabled
    - Attention map uses lightweight OpenCV processing
    - Aggressive cleanup is used
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
# BASE PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)


app.config["UPLOAD_FOLDER"] = (
    UPLOAD_FOLDER
)


# Maximum upload size = 10 MB

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
# ALLOWED FILES
# ============================================================

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png",
    "webp"
}


# ============================================================
# DATABASE
# ============================================================

try:

    create_tables()

    print(
        "Database tables initialized."
    )

except Exception as e:

    print(
        "Database initialization error:",
        e
    )


# ============================================================
# MODEL INITIALIZATION
# ============================================================

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

    MODEL = None

    TRANSFORM = None


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
# CLEANUP
# ============================================================

def safe_cleanup():

    try:

        cleanup()

    except Exception as e:

        print(
            "Cleanup warning:",
            e
        )

    try:

        gc.collect()

    except Exception:

        pass


# ============================================================
# UNIQUE FILE NAME
# ============================================================

def create_unique_filename(
    original_filename
):

    safe_name = secure_filename(
        original_filename
    )

    if not safe_name:

        safe_name = "image.jpg"

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
            False,

        "ai_attention_map":
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
# FALLBACK DISEASE INFO
# ============================================================

def fallback_disease_info(
    prediction
):

    return {

        "plant": "Unknown",

        "disease": prediction,

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
    prediction
):

    severity_level = (
        "Not calculated"
    )

    affected_area = 0


    try:

        result = estimate_severity(
            filepath
        )


        if isinstance(
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
            dict
        ):

            severity_level = result.get(

                "severity",

                result.get(
                    "severity_level",
                    "Not calculated"
                )

            )

            affected_area = result.get(

                "affected_area",

                result.get(
                    "area",
                    0
                )

            )


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

        try:

            result = estimate_severity(

                filepath,

                prediction

            )


            if isinstance(
                result,
                tuple
            ):

                if len(result) >= 2:

                    severity_level = result[0]

                    affected_area = result[1]


            elif isinstance(
                result,
                dict
            ):

                severity_level = result.get(

                    "severity",

                    result.get(
                        "severity_level",
                        "Not calculated"
                    )

                )

                affected_area = result.get(

                    "affected_area",

                    0

                )


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


    try:

        affected_area = float(
            affected_area
        )

    except Exception:

        affected_area = 0


    affected_area = max(

        0,

        min(
            100,
            affected_area
        )

    )


    if not severity_level:

        severity_level = (
            "Not calculated"
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

            return ""

        return result


    except TypeError:

        try:

            result = get_severity_advice(
                severity
            )

            if result is None:

                return ""

            return result


        except Exception as e:

            print(
                "Severity advice error:",
                e
            )

            return ""


    except Exception as e:

        print(
            "Severity advice error:",
            e
        )

        return ""


# ============================================================
# AI HIGHLIGHT
# ============================================================

def generate_highlight_safe(
    input_path,
    filename
):

    """
    AI HIGHLIGHT IS INTENTIONALLY DISABLED.

    We do NOT call create_highlight().

    This avoids:
        - extra image processing
        - duplicate visualization
        - memory usage
        - previous create_highlight errors

    Returns:
        None
    """

    print(
        "AI Highlight disabled."
    )

    return None


# ============================================================
# LIGHTWEIGHT ATTENTION MAP
# ============================================================

def generate_attention_safe(
    input_path,
    filename
):

    """
    Lightweight browser-safe / Render-safe attention map.

    IMPORTANT:

    This is NOT Grad-CAM.

    It does not load another AI model.

    It uses OpenCV image analysis to create a visual
    attention-style map around non-uniform regions of
    the plant image.

    This is intentionally lightweight because Render
    free instances have approximately 512 MB RAM.
    """

    image = None

    hsv = None

    gray = None

    edges = None

    attention = None

    heatmap = None

    overlay = None

    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        image = cv2.imread(
            input_path
        )

        if image is None:

            print(
                "Attention map: image could not be read."
            )

            return None


        # ====================================================
        # RESIZE
        # ====================================================

        height, width = image.shape[:2]


        max_dimension = 700


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
        # LEAF MASK
        # ====================================================

        lower_green = np.array(

            [25, 30, 30],

            dtype=np.uint8

        )


        upper_green = np.array(

            [100, 255, 255],

            dtype=np.uint8

        )


        leaf_mask = cv2.inRange(

            hsv,

            lower_green,

            upper_green

        )


        # ====================================================
        # GRAYSCALE
        # ====================================================

        gray = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2GRAY

        )


        # ====================================================
        # EDGE INFORMATION
        # ====================================================

        edges = cv2.Canny(

            gray,

            50,

            150

        )


        # ====================================================
        # NON-GREEN / TEXTURE INFORMATION
        # ====================================================

        non_green = cv2.bitwise_not(
            leaf_mask
        )


        # ====================================================
        # COMBINE ATTENTION SIGNALS
        # ====================================================

        attention = cv2.addWeighted(

            edges,

            0.55,

            non_green,

            0.45,

            0

        )


        # ====================================================
        # BLUR
        # ====================================================

        attention = cv2.GaussianBlur(

            attention,

            (
                21,
                21
            ),

            0

        )


        # ====================================================
        # NORMALIZE
        # ====================================================

        min_value = float(
            attention.min()
        )

        max_value = float(
            attention.max()
        )


        if max_value > min_value:

            attention = cv2.normalize(

                attention,

                None,

                0,

                255,

                cv2.NORM_MINMAX

            )

        else:

            attention[:] = 0


        attention = np.uint8(
            attention
        )


        # ====================================================
        # HEATMAP
        # ====================================================

        heatmap = cv2.applyColorMap(

            attention,

            cv2.COLORMAP_JET

        )


        # ====================================================
        # OVERLAY
        # ====================================================

        overlay = cv2.addWeighted(

            image,

            0.55,

            heatmap,

            0.45,

            0

        )


        # ====================================================
        # OUTPUT NAME
        # ====================================================

        base_name = os.path.splitext(
            filename
        )[0]


        attention_filename = (

            f"{base_name}_attention.jpg"

        )


        output_path = os.path.join(

            app.config[
                "UPLOAD_FOLDER"
            ],

            attention_filename

        )


        # ====================================================
        # SAVE
        # ====================================================

        success = cv2.imwrite(

            output_path,

            overlay,

            [
                cv2.IMWRITE_JPEG_QUALITY,
                85
            ]

        )


        if not success:

            print(
                "Attention map save failed."
            )

            return None


        print(
            "Attention map created:",
            attention_filename
        )


        return attention_filename


    except Exception as e:

        print(
            "Attention map generation failed:",
            e
        )

        traceback.print_exc()

        return None


    finally:

        # ====================================================
        # RELEASE NUMPY / OPENCV MEMORY
        # ====================================================

        image = None

        hsv = None

        gray = None

        edges = None

        attention = None

        heatmap = None

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


        if isinstance(
            result,
            str
        ):

            if os.path.exists(result):

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


        if os.path.exists(
            report_path
        ):

            return report_filename


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
# MAIN UPLOAD
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
    # VALIDATE FILE
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
    # SAVE
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
            f"Confidence: "
            f"{confidence:.2f}%"
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


    if info is None:

        info = fallback_disease_info(
            prediction
        )


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

    # INTENTIONALLY DISABLED

    highlight_name = None


    print(
        "AI Highlight: DISABLED"
    )


    # ========================================================
    # AI ATTENTION MAP
    # ========================================================

    attention_name = (

        generate_attention_safe(

            filepath,

            filename

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
    # FINAL CLEANUP
    # ========================================================

    safe_cleanup()


    # ========================================================
    # RESULT
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
        "PDF:",
        report_name
    )

    print("=" * 60)


    return render_template(

        "result.html",

        image=filename,

        # Highlight disabled
        highlight=None,

        # Attention map enabled
        gradcam=attention_name,

        attention=attention_name,

        confidence=round(
            confidence,
            2
        ),

        confidence_status=
            confidence_status,

        top_predictions=
            top_predictions,

        severity=
            severity_level,

        affected_area=
            affected_area,

        info=
            info,

        explanation=
            explanation,

        severity_advice=
            severity_advice,

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
    # CHECK FILE
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

    except Exception:

        info = None


    if info is None:

        info = fallback_disease_info(
            prediction
        )


    # ========================================================
    # LOW CONFIDENCE
    # ========================================================

    if confidence < 40:

        return jsonify({

            "success": True,

            "recognized": False,

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
    # ADVICE
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
    # LIGHTWEIGHT ATTENTION MAP
    # ========================================================

    attention_name = (

        generate_attention_safe(

            filepath,

            filename

        )

    )


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

        # Highlight intentionally disabled
        "highlight":
            None,

        # Attention enabled
        "attention":
            attention_name

    }


    safe_cleanup()


    return jsonify(
        response
    )


# ============================================================
# HISTORY
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
# UPLOADED FILES
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


    app.run(

        host="0.0.0.0",

        port=port,

        debug=False

    )