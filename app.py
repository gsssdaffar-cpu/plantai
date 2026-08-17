# -*- coding: utf-8 -*-

"""
============================================================
PlantAI Flask Application
============================================================

Render optimized version.

Features:
    - Plant disease prediction
    - Disease information
    - Severity estimation
    - AI explanation
    - AI highlight
    - AI attention map / Grad-CAM
    - PDF report
    - Prediction history
    - Dashboard
    - Chatbot
    - Mobile API

Important for Render:
    - Gunicorn workers should remain 1
    - Prediction model is loaded once by predict.py
    - cleanup() is called after predictions
    - Expensive optional features are isolated
    - Failure of highlight / explanation / PDF does not
      destroy the main diagnosis
============================================================
"""

import os
import gc
import uuid
import traceback

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    jsonify
)

from werkzeug.utils import secure_filename


# ============================================================
# AI / PROJECT IMPORTS
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

from highlight import (
    create_highlight
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
# BASIC CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

REPORT_FOLDER = UPLOAD_FOLDER

HIGHLIGHT_FOLDER = UPLOAD_FOLDER

ATTENTION_FOLDER = UPLOAD_FOLDER


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)

os.makedirs(
    HIGHLIGHT_FOLDER,
    exist_ok=True
)

os.makedirs(
    ATTENTION_FOLDER,
    exist_ok=True
)


# ============================================================
# ALLOWED IMAGE EXTENSIONS
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
# UTILITY FUNCTIONS
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
# SAFE CLEANUP
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
# SAFE FILE NAME
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

    return {

        "status": "ok",

        "service": "PlantAI",

        "model_loaded":
            MODEL is not None

    }


# ============================================================
# CHATBOT
# ============================================================

@app.route(
    "/chatbot",
    methods=[
        "GET",
        "POST"
    ]
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
# DISEASE FALLBACK
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

    """
    Try the project's severity estimator.

    Different versions of severity.py may have different
    function signatures, so this function safely tries the
    common forms.

    Returns:

        severity_level
        affected_area
    """

    severity_level = "Not calculated"

    affected_area = 0


    try:

        # ----------------------------------------------------
        # First attempt
        # ----------------------------------------------------

        result = estimate_severity(
            filepath
        )


        if isinstance(
            result,
            tuple
        ):

            if len(result) >= 2:

                severity_level = (
                    result[0]
                )

                affected_area = (
                    result[1]
                )

            elif len(result) == 1:

                severity_level = (
                    result[0]
                )


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


        print(
            "Severity:",
            severity_level
        )

        print(
            "Affected area:",
            affected_area
        )


    except TypeError:

        # ----------------------------------------------------
        # Alternative signature
        # ----------------------------------------------------

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

                    severity_level = (
                        result[0]
                    )

                    affected_area = (
                        result[1]
                    )


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


    # --------------------------------------------------------
    # Normalize affected area
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Normalize severity
    # --------------------------------------------------------

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
    Generate AI-highlight image.

    IMPORTANT:
    create_highlight() in your project requires:

        input_path
        output_path

    Therefore we explicitly provide both.
    """

    try:

        base_name = os.path.splitext(
            filename
        )[0]

        extension = os.path.splitext(
            filename
        )[1]

        highlight_filename = (
            f"{base_name}_highlight"
            f"{extension}"
        )

        output_path = os.path.join(

            app.config["UPLOAD_FOLDER"],

            highlight_filename

        )


        print(
            "Creating AI highlight..."
        )

        print(
            "Input:",
            input_path
        )

        print(
            "Output:",
            output_path
        )


        result = create_highlight(

            input_path,

            output_path

        )


        # ----------------------------------------------------
        # Function may return output path
        # ----------------------------------------------------

        if isinstance(
            result,
            str
        ):

            if os.path.exists(result):

                return os.path.basename(
                    result
                )


        # ----------------------------------------------------
        # Our expected output
        # ----------------------------------------------------

        if os.path.exists(
            output_path
        ):

            return highlight_filename


        print(
            "Highlight was not created."
        )

        return None


    except Exception as e:

        print(
            "Highlight generation failed:",
            e
        )

        traceback.print_exc()

        return None


# ============================================================
# AI ATTENTION MAP
# ============================================================

def generate_attention_safe(
    input_path,
    filename
):

    """
    Try the project's highlight module for an attention map.

    If your highlight.py exposes a separate attention/Grad-CAM
    function, this function attempts to use it.

    If unavailable, returns None instead of crashing prediction.
    """

    try:

        import highlight as highlight_module


        function_names = [

            "create_attention_map",

            "create_attention",

            "generate_attention_map",

            "generate_attention",

            "create_gradcam",

            "generate_gradcam",

            "create_grad_cam",

            "generate_grad_cam"

        ]


        attention_function = None


        for function_name in function_names:

            candidate = getattr(

                highlight_module,

                function_name,

                None

            )

            if callable(candidate):

                attention_function = candidate

                break


        if attention_function is None:

            print(
                "No attention-map function found "
                "in highlight.py."
            )

            return None


        base_name = os.path.splitext(
            filename
        )[0]

        extension = os.path.splitext(
            filename
        )[1]


        attention_filename = (
            f"{base_name}_attention"
            f"{extension}"
        )


        output_path = os.path.join(

            app.config["UPLOAD_FOLDER"],

            attention_filename

        )


        print(
            "Creating AI attention map..."
        )


        # ----------------------------------------------------
        # Try input + output
        # ----------------------------------------------------

        try:

            result = attention_function(

                input_path,

                output_path

            )


        except TypeError:

            # ------------------------------------------------
            # Try only input
            # ------------------------------------------------

            result = attention_function(
                input_path
            )


        # ----------------------------------------------------
        # Function returned path
        # ----------------------------------------------------

        if isinstance(
            result,
            str
        ):

            if os.path.exists(result):

                return os.path.basename(
                    result
                )


        # ----------------------------------------------------
        # Expected output
        # ----------------------------------------------------

        if os.path.exists(
            output_path
        ):

            return attention_filename


        print(
            "Attention map was not created."
        )

        return None


    except Exception as e:

        print(
            "Attention map generation failed:",
            e
        )

        traceback.print_exc()

        return None


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

    """
    Generate PDF report safely.

    Failure to generate PDF must NOT make prediction fail.
    """

    try:

        base_name = os.path.splitext(
            filename
        )[0]

        report_filename = (
            f"{base_name}_report.pdf"
        )


        report_path = os.path.join(

            app.config["UPLOAD_FOLDER"],

            report_filename

        )


        print(
            "Creating PDF report..."
        )


        # ----------------------------------------------------
        # Attempt common report_generator signatures
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # If function returns a path
        # ----------------------------------------------------

        if isinstance(
            result,
            str
        ):

            if os.path.exists(result):

                return os.path.basename(
                    result
                )


        # ----------------------------------------------------
        # If function returned a pathlib path
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Expected output path
        # ----------------------------------------------------

        if os.path.exists(
            report_path
        ):

            return report_filename


        # ----------------------------------------------------
        # Search for generated PDF
        # ----------------------------------------------------

        for candidate in os.listdir(
            app.config["UPLOAD_FOLDER"]
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
# SAVE HISTORY
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
    # CHECK EXTENSION
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
    # UNIQUE FILE NAME
    # ========================================================

    filename = create_unique_filename(
        file.filename
    )


    filepath = os.path.join(

        app.config["UPLOAD_FOLDER"],

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
            f"Confidence: "
            f"{confidence:.2f}%"
        )

        print(
            "Status:",
            confidence_status
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

        # Release prediction tensors

        safe_cleanup()


    # ========================================================
    # DISEASE LOOKUP
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

        print(
            "Disease information missing:"
            ,
            prediction
        )

        info = fallback_disease_info(
            prediction
        )


    # ========================================================
    # LOW CONFIDENCE
    # ========================================================

    if confidence < 40:

        try:

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

        finally:

            safe_cleanup()


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

    highlight_name = (
        generate_highlight_safe(

            filepath,

            filename

        )
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
    # SAVE HISTORY
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
    # PDF REPORT
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
    # RESULT PAGE
    # ========================================================

    print()
    print(
        "Rendering result page..."
    )

    print(
        "Highlight:",
        highlight_name
    )

    print(
        "Attention:",
        attention_name
    )

    print(
        "Report:",
        report_name
    )

    print("=" * 60)


    return render_template(

        "result.html",

        image=filename,

        highlight=highlight_name,

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
    # CHECK EXTENSION
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
    # UNIQUE FILE
    # ========================================================

    filename = create_unique_filename(
        file.filename
    )


    filepath = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    try:

        file.save(
            filepath
        )


    except Exception as e:

        print(
            "File save error:",
            e
        )

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
    # DISEASE LOOKUP
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
            top_predictions

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

        app.config["UPLOAD_FOLDER"],

        filename

    )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(
    413
)
def file_too_large(error):

    return (

        "Image is too large. "
        "Maximum upload size is 10 MB.",

        413

    )


@app.errorhandler(
    500
)
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
# RUN LOCAL DEVELOPMENT
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