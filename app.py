import os

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory
)

from werkzeug.utils import secure_filename

from PIL import Image

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

from gradcam import create_gradcam

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

create_tables()


# ============================================================
# UPLOAD CONFIGURATION
# ============================================================

UPLOAD_FOLDER = "uploads"

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

    stats = dashboard_stats()

    recent = recent_predictions()

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

            answer = chatbot_response(
                question
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
#
# THIS IS YOUR EXISTING STABLE WEB ROUTE.
# DO NOT CHANGE ITS BEHAVIOR.
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    # --------------------------------------------------------
    # Check uploaded file
    # --------------------------------------------------------

    if "image" not in request.files:

        return "No image uploaded."


    file = request.files["image"]


    if file.filename == "":

        return "No file selected."


    # --------------------------------------------------------
    # Secure filename
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )


    if not filename:

        return "Invalid file name."


    # --------------------------------------------------------
    # Save uploaded image
    # --------------------------------------------------------

    filepath = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )


    file.save(
        filepath
    )


    # ========================================================
    # FILENAMES
    # ========================================================

    gradcam_name = (
        "gradcam_"
        + filename
    )


    gradcam_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        gradcam_name

    )


    # ========================================================
    # AI PREDICTION
    # ========================================================

    prediction, confidence, top_predictions, confidence_status = (
        predict_image(
            filepath
        )
    )


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
        "===================================="
    )


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image = Image.open(
        filepath
    ).convert(
        "RGB"
    )


    # ========================================================
    # IMAGE TRANSFORM
    # ========================================================

    tensor = get_transform()(
        image
    )


    tensor = tensor.unsqueeze(
        0
    )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    try:

        create_gradcam(

            get_model(),

            tensor,

            filepath,

            gradcam_path

        )

    except Exception as e:

        print(
            "Grad-CAM error:",
            e
        )

        gradcam_name = None


    # ========================================================
    # UNKNOWN / LOW CONFIDENCE
    # ========================================================

    if confidence < 40:

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


    # ========================================================
    # SEVERITY ANALYSIS
    # ========================================================

    severity = estimate_severity(
        filepath
    )


    severity_level = (
        severity.get(
            "level",
            "Unknown"
        )
    )


    affected_area = (
        severity.get(
            "area",
            0
        )
    )


    # ========================================================
    # AI EXPLANATION
    # ========================================================

    explanation = generate_explanation(

        info,

        confidence,

        severity_level,

        affected_area

    )


    # ========================================================
    # SEVERITY ADVICE
    # ========================================================

    severity_advice = (
        get_severity_advice(
            severity_level
        )
    )


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
    # CREATE PDF REPORT
    # ========================================================

    report_name = (
        "PlantAI_Report.pdf"
    )


    report_path = os.path.join(

        app.config["UPLOAD_FOLDER"],

        report_name

    )


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
# NEW ENDPOINT
#
# POST /api/predict
#
# This does NOT replace /upload.
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def api_predict():

    print()
    print("=" * 60)
    print("🌱 MOBILE API REQUEST")
    print("=" * 60)


    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if "image" not in request.files:

        print("ERROR: No image uploaded")

        return {

            "success": False,

            "error": "No image uploaded"

        }, 400


    file = request.files["image"]


    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if file.filename == "":

        print("ERROR: No file selected")

        return {

            "success": False,

            "error": "No file selected"

        }, 400


    # --------------------------------------------------------
    # Secure filename
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )


    if not filename:

        print("ERROR: Invalid filename")

        return {

            "success": False,

            "error": "Invalid file name"

        }, 400


    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

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

        prediction, confidence, top_predictions, confidence_status = (
            predict_image(
                filepath
            )
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


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

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

        print(
            "Low confidence result."
        )

        print(
            "=" * 60
        )


        return {

            "success": True,

            "recognized": False,

            "message":
                "Plant could not be identified confidently.",

            "prediction": prediction,

            "plant": info["plant"],

            "disease": info["disease"],

            "confidence": round(
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

        "success": True,

        "recognized": True,

        "image": filename,

        "prediction": prediction,

        "plant": info["plant"],

        "disease": info["disease"],

        "confidence": round(
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


    print()
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


    rows = get_history(
        search
    )


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
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )