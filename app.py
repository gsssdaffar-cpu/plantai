import os

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

from werkzeug.utils import secure_filename


# ============================================================
# AI / PROJECT IMPORTS
# ============================================================

from predict import (
    predict_image,
    get_model,
    get_transform
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

from highlight import create_highlight

from report_generator import create_report

from database import create_tables

from chatbot import chatbot_response

from explanation import generate_explanation


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# DATABASE
# ============================================================

try:

    create_tables()

    print("Database tables initialized.")

except Exception as e:

    print(
        "Database initialization error:",
        e
    )


# ============================================================
# UPLOAD CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
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
# WEB UPLOAD PAGE
# ============================================================

@app.route("/upload")
def upload_page():

    return render_template(
        "upload.html"
    )


# ============================================================
# WEB UPLOAD + AI PREDICTION
#
# LIGHTWEIGHT VERSION FOR RENDER
#
# IMPORTANT:
#
# Grad-CAM       -> DISABLED
# Severity       -> DISABLED
# Highlight      -> DISABLED
# Explanation    -> DISABLED
# PDF            -> DISABLED
#
# This keeps the request lightweight and avoids
# Render 502 / worker timeout / memory problems.
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
    # CHECK IMAGE
    # ========================================================

    if "image" not in request.files:

        print(
            "ERROR: No image uploaded."
        )

        return (
            "No image uploaded.",
            400
        )


    file = request.files["image"]


    # ========================================================
    # CHECK FILE NAME
    # ========================================================

    if file.filename == "":

        print(
            "ERROR: No file selected."
        )

        return (
            "No file selected.",
            400
        )


    # ========================================================
    # SECURE FILE NAME
    # ========================================================

    filename = secure_filename(
        file.filename
    )


    if not filename:

        print(
            "ERROR: Invalid filename."
        )

        return (
            "Invalid file name.",
            400
        )


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    filepath = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )


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
    # AI PREDICTION
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


        return (

            f"Prediction failed: {str(e)}",

            500

        )


    # ========================================================
    # LOW CONFIDENCE
    # ========================================================

    if confidence < 40:

        print(
            "Low confidence prediction."
        )


        return render_template(

            "unknown.html",

            image=filename,

            confidence=round(

                confidence,

                2

            )

        )


    # ========================================================
    # DISEASE DATABASE LOOKUP
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
    # FALLBACK DISEASE INFORMATION
    # ========================================================

    if info is None:

        print(
            "WARNING: No disease information found:",
            prediction
        )


        info = {

            "plant": "Unknown",

            "disease": prediction,

            "cause":
                "Information not available",

            "symptoms": [],

            "treatment": [],

            "organic_treatment": [],

            "prevention": []

        }


    # ========================================================
    # DISABLE EXPENSIVE FEATURES
    #
    # THESE ARE INTENTIONALLY DISABLED FOR RENDER
    # ========================================================


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    gradcam_name = None

    print(
        "Grad-CAM skipped."
    )


    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    severity_level = "Not calculated"

    affected_area = 0

    print(
        "Severity analysis skipped."
    )


    # --------------------------------------------------------
    # AI Explanation
    # --------------------------------------------------------

    explanation = ""

    print(
        "AI explanation skipped."
    )


    # --------------------------------------------------------
    # Severity Advice
    # --------------------------------------------------------

    severity_advice = ""

    print(
        "Severity advice skipped."
    )


    # --------------------------------------------------------
    # Highlight
    # --------------------------------------------------------

    highlight_name = None

    print(
        "Highlight generation skipped."
    )


    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    report_name = None

    print(
        "PDF generation skipped."
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

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

            severity_level,

            affected_area

        )


        print(
            "History saved."
        )


    except Exception as e:

        print(
            "History save error:",
            e
        )


    # ========================================================
    # RESULT PAGE
    # ========================================================

    print(
        "Rendering result page..."
    )


    return render_template(

        "result.html",

        image=filename,

        highlight=highlight_name,

        gradcam=gradcam_name,

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
#
# POST /api/predict
#
# This remains available for mobile/API clients.
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    print()
    print("=" * 60)
    print("PLANTAI MOBILE API REQUEST")
    print("=" * 60)


    # ========================================================
    # CHECK IMAGE
    # ========================================================

    if "image" not in request.files:

        print(
            "ERROR: No image uploaded."
        )


        return {

            "success": False,

            "error":
                "No image uploaded"

        }, 400


    file = request.files["image"]


    # ========================================================
    # CHECK FILE NAME
    # ========================================================

    if file.filename == "":

        print(
            "ERROR: No file selected."
        )


        return {

            "success": False,

            "error":
                "No file selected"

        }, 400


    # ========================================================
    # SECURE FILE NAME
    # ========================================================

    filename = secure_filename(
        file.filename
    )


    if not filename:

        print(
            "ERROR: Invalid filename."
        )


        return {

            "success": False,

            "error":
                "Invalid file name"

        }, 400


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    filepath = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )


    try:

        file.save(
            filepath
        )

    except Exception as e:

        print(
            "ERROR saving image:",
            e
        )


        return {

            "success": False,

            "error":
                "Could not save image"

        }, 500


    print(
        "Image:",
        filename
    )


    # ========================================================
    # AI PREDICTION
    # ========================================================

    try:

        print(
            "Starting prediction..."
        )


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


        return {

            "success": False,

            "error":
                "AI prediction failed",

            "details":
                str(e)

        }, 500


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


    # ========================================================
    # DISEASE DATABASE
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

        info = {

            "plant": "Unknown",

            "disease": prediction,

            "cause":
                "Information not available",

            "symptoms": [],

            "treatment": [],

            "organic_treatment": [],

            "prevention": []

        }


    # ========================================================
    # LOW CONFIDENCE
    # ========================================================

    if confidence < 40:

        return {

            "success": True,

            "recognized": False,

            "message":
                "Plant could not be identified confidently.",

            "prediction":
                prediction,

            "plant":
                info["plant"],

            "disease":
                info["disease"],

            "confidence":
                round(
                    confidence,
                    2
                ),

            "confidence_status":
                confidence_status,

            "top_predictions":
                top_predictions

        }


    # ========================================================
    # LIGHTWEIGHT API
    #
    # Do not run expensive image processing here either.
    # ========================================================

    severity_level = "Not calculated"

    affected_area = 0

    explanation = ""

    severity_advice = ""


    # ========================================================
    # SAVE HISTORY
    # ========================================================

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

            severity_level,

            affected_area

        )

    except Exception as e:

        print(
            "History save error:",
            e
        )


    # ========================================================
    # API RESPONSE
    # ========================================================

    response = {

        "success": True,

        "recognized": True,

        "image":
            filename,

        "prediction":
            prediction,

        "plant":
            info["plant"],

        "disease":
            info["disease"],

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
            info["cause"],

        "symptoms":
            info["symptoms"],

        "treatment":
            info["treatment"],

        "organic_treatment":
            info["organic_treatment"],

        "prevention":
            info["prevention"],

        "explanation":
            explanation,

        "severity_advice":
            severity_advice,

        "top_predictions":
            top_predictions

    }


    print(
        "Mobile API response prepared."
    )


    print(
        "=" * 60
    )


    return response


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
    "/uploads/<filename>"
)
def uploaded_file(
    filename
):

    return send_from_directory(

        app.config["UPLOAD_FOLDER"],

        filename

    )


# ============================================================
# HEALTH CHECK
#
# Useful for Render.
# ============================================================

@app.route("/health")
def health():

    return {

        "status": "ok",

        "service": "PlantAI"

    }


# ============================================================
# RUN APPLICATION
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