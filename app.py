import os
import time

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

from werkzeug.utils import secure_filename

from predict import predict_image

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

    print("Database initialization error:", e)


# ============================================================
# PATH CONFIGURATION
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
# WEB UPLOAD + AI DIAGNOSIS
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    request_start = time.time()

    print()
    print("=" * 60)
    print("WEB UPLOAD REQUEST STARTED")
    print("=" * 60)


    # ========================================================
    # CHECK FILE
    # ========================================================

    if "image" not in request.files:

        print(
            "ERROR: No image uploaded."
        )

        return "No image uploaded.", 400


    file = request.files["image"]


    if file.filename == "":

        print(
            "ERROR: No file selected."
        )

        return "No file selected.", 400


    # ========================================================
    # SECURE FILENAME
    # ========================================================

    filename = secure_filename(
        file.filename
    )


    if not filename:

        print(
            "ERROR: Invalid filename."
        )

        return "Invalid file name.", 400


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
            "File save error:",
            e
        )

        return "Could not save image.", 500


    print(
        "Image saved:",
        filepath
    )


    print(
        "Upload save time:",
        round(
            time.time() - request_start,
            2
        ),
        "seconds"
    )


    # ========================================================
    # AI PREDICTION
    # ========================================================

    prediction_start = time.time()


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

        return (
            "AI prediction failed.",
            500
        )


    print()
    print(
        "===================================="
    )

    print(
        "AI Prediction:",
        prediction
    )

    print(
        f"AI Confidence: {confidence:.2f}%"
    )

    print(
        "AI Status:",
        confidence_status
    )

    print(
        "Prediction time:",
        round(
            time.time() - prediction_start,
            2
        ),
        "seconds"
    )

    print(
        "===================================="
    )


    # ========================================================
    # GRAD-CAM
    #
    # DISABLED FOR RENDER
    #
    # Grad-CAM performs additional neural-network operations
    # and was causing the Render worker to timeout / run out
    # of memory.
    # ========================================================

    gradcam_name = None

    print(
        "Grad-CAM skipped."
    )


    # ========================================================
    # UNKNOWN / LOW CONFIDENCE
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

    info = get_disease(
        prediction
    )


    # ========================================================
    # FALLBACK INFORMATION
    # ========================================================

    if info is None:

        print(
            "WARNING: No disease information found for:",
            prediction
        )


        info = {

            "plant": "Unknown",

            "disease": prediction,

            "cause": "Information not available",

            "symptoms": [],

            "treatment": [],

            "organic_treatment": [],

            "prevention": []

        }


    print(
        "Disease lookup completed in",
        round(
            time.time() - request_start,
            2
        ),
        "seconds"
    )


    # ========================================================
    # SEVERITY ANALYSIS
    # ========================================================

    severity_start = time.time()


    try:

        severity = estimate_severity(
            filepath
        )


        severity_level = severity.get(
            "level",
            "Unknown"
        )


        affected_area = severity.get(
            "area",
            0
        )


    except Exception as e:

        print(
            "Severity error:",
            e
        )

        severity_level = "Unknown"

        affected_area = 0


    print(
        "Severity completed in",
        round(
            time.time() - severity_start,
            2
        ),
        "seconds"
    )


    # ========================================================
    # AI EXPLANATION
    # ========================================================

    explanation_start = time.time()


    try:

        explanation = generate_explanation(

            info,

            confidence,

            severity_level,

            affected_area

        )

    except Exception as e:

        print(
            "Explanation error:",
            e
        )

        explanation = ""


    print(
        "Explanation completed in",
        round(
            time.time() - explanation_start,
            2
        ),
        "seconds"
    )


    # ========================================================
    # SEVERITY ADVICE
    # ========================================================

    try:

        severity_advice = get_severity_advice(
            severity_level
        )

    except Exception as e:

        print(
            "Severity advice error:",
            e
        )

        severity_advice = ""


    # ========================================================
    # HIGHLIGHT INFECTED AREA
    # ========================================================

    highlight_name = (
        "highlight_"
        + filename
    )


    highlight_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        highlight_name

    )


    highlight_start = time.time()


    try:

        create_highlight(

            filepath,

            highlight_path

        )

    except Exception as e:

        print(
            "Highlight generation error:",
            e
        )

        highlight_name = None


    print(
        "Highlight completed in",
        round(
            time.time() - highlight_start,
            2
        ),
        "seconds"
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    try:

        save_prediction(

            filename,

            info["plant"],

            info["disease"],

            confidence,

            severity_level,

            affected_area

        )

        print(
            "Prediction history saved."
        )

    except Exception as e:

        print(
            "History save error:",
            e
        )


    # ========================================================
    # CREATE PDF REPORT
    # ========================================================

    report_name = (
        "PlantAI_Report.pdf"
    )


    report_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        report_name

    )


    report_start = time.time()


    try:

        create_report(

            report_path,

            info["plant"],

            info["disease"],

            confidence,

            severity_level,

            affected_area,

            info["cause"],

            info["symptoms"],

            info["treatment"],

            info["organic_treatment"],

            info["prevention"]

        )

    except Exception as e:

        print(
            "PDF report error:",
            e
        )

        report_name = None


    print(
        "PDF generation completed in",
        round(
            time.time() - report_start,
            2
        ),
        "seconds"
    )


    # ========================================================
    # TOTAL PROCESSING TIME
    # ========================================================

    total_time = round(
        time.time() - request_start,
        2
    )


    print()
    print("=" * 60)
    print(
        "TOTAL WEB REQUEST TIME:",
        total_time,
        "seconds"
    )
    print("=" * 60)


    # ========================================================
    # RESULT PAGE
    # ========================================================

    return render_template(

        "result.html",

        image=filename,

        highlight=highlight_name,

        gradcam=gradcam_name,

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

        severity_advice=severity_advice,

        report=report_name

    )


# ============================================================
# MOBILE API
#
# POST /api/predict
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    request_start = time.time()

    print()
    print("=" * 60)
    print("MOBILE API REQUEST")
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

            "error": "No image uploaded"

        }, 400


    file = request.files["image"]


    if file.filename == "":

        print(
            "ERROR: No file selected."
        )

        return {

            "success": False,

            "error": "No file selected"

        }, 400


    # ========================================================
    # SECURE FILENAME
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

            "error": "Invalid file name"

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

            "error": "Could not save image"

        }, 500


    print(
        "Image:",
        filename
    )


    # ========================================================
    # AI PREDICTION
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

        return {

            "success": False,

            "error": "AI prediction failed",

            "details": str(e)

        }, 500


    print(
        "Prediction:",
        prediction
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print(
        "Status:",
        confidence_status
    )


    # ========================================================
    # DISEASE DATABASE
    # ========================================================

    info = get_disease(
        prediction
    )


    if info is None:

        print(
            "WARNING: Disease information not found:",
            prediction
        )


        info = {

            "plant": "Unknown",

            "disease": prediction,

            "cause": "Information not available",

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
    # SEVERITY
    # ========================================================

    try:

        severity = estimate_severity(
            filepath
        )


        severity_level = severity.get(
            "level",
            "Unknown"
        )


        affected_area = severity.get(
            "area",
            0
        )


    except Exception as e:

        print(
            "Severity error:",
            e
        )

        severity_level = "Unknown"

        affected_area = 0


    # ========================================================
    # AI EXPLANATION
    # ========================================================

    try:

        explanation = generate_explanation(

            info,

            confidence,

            severity_level,

            affected_area

        )

    except Exception as e:

        print(
            "Explanation error:",
            e
        )

        explanation = ""


    # ========================================================
    # SEVERITY ADVICE
    # ========================================================

    try:

        severity_advice = get_severity_advice(

            severity_level

        )

    except Exception as e:

        print(
            "Severity advice error:",
            e
        )

        severity_advice = ""


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    try:

        save_prediction(

            filename,

            info["plant"],

            info["disease"],

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
    # MOBILE RESPONSE
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
        "Mobile API completed in",
        round(
            time.time() - request_start,
            2
        ),
        "seconds"
    )


    print("=" * 60)


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