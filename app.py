# -*- coding: utf-8 -*-

import os
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

from gradcam import create_gradcam

from report_generator import create_report

from database import create_tables

from chatbot import chatbot_response

from explanation import generate_explanation


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# CONFIGURATION
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
# LOAD MODEL
#
# Model is already loaded by predict.py.
# We retrieve it here for Grad-CAM.
# ============================================================

try:

    AI_MODEL = get_model()

    AI_TRANSFORM = get_transform()

    print(
        "AI model ready for application."
    )

except Exception as e:

    AI_MODEL = None

    AI_TRANSFORM = None

    print(
        "WARNING: Could not initialize AI model:",
        e
    )


# ============================================================
# SAFE FUNCTION CALL
# ============================================================

def safe_call(
    function,
    **kwargs
):

    try:

        return function(
            **kwargs
        )

    except Exception as e:

        print()
        print(
            "=" * 60
        )

        print(
            f"ERROR in {function.__name__}"
        )

        print(
            str(e)
        )

        print(
            "=" * 60
        )

        traceback.print_exc()

        return None


# ============================================================
# CREATE UNIQUE FILE NAME
# ============================================================

def make_output_filename(
    original_filename,
    suffix
):

    base_name = os.path.splitext(
        secure_filename(
            original_filename
        )
    )[0]

    unique_id = uuid.uuid4().hex[:10]

    return (
        f"{base_name}_"
        f"{unique_id}_"
        f"{suffix}.jpg"
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
# UPLOAD PAGE
# ============================================================

@app.route("/upload")
def upload_page():

    return render_template(
        "upload.html"
    )


# ============================================================
# MAIN WEB UPLOAD
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload():

    print()
    print("=" * 70)
    print("🌱 PLANTAI WEB UPLOAD")
    print("=" * 70)


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
    # SAVE ORIGINAL IMAGE
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
            "Original image saved:",
            filepath
        )

    except Exception as e:

        print(
            "File save error:",
            e
        )

        traceback.print_exc()

        return (
            "Could not save image.",
            500
        )


    # ========================================================
    # AI PREDICTION
    # ========================================================

    print()
    print(
        "Starting AI prediction..."
    )


    try:

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

        return (

            f"Prediction failed: {str(e)}",

            500

        )


    # ========================================================
    # DISEASE DATABASE LOOKUP
    # ========================================================

    print()
    print(
        "Looking up disease information..."
    )


    try:

        info = get_disease(
            prediction
        )

    except Exception as e:

        print(
            "Disease lookup error:",
            e
        )

        traceback.print_exc()

        info = None


    # ========================================================
    # FALLBACK DISEASE INFORMATION
    # ========================================================

    if info is None:

        print(
            "WARNING: Disease information not found:",
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


    print(
        "Plant:",
        info.get(
            "plant",
            "Unknown"
        )
    )

    print(
        "Disease:",
        info.get(
            "disease",
            prediction
        )
    )


    # ========================================================
    # LOW CONFIDENCE
    # ========================================================

    if confidence < 40:

        print()
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
    # ========================================================
    # AI HIGHLIGHT
    # ========================================================
    # ========================================================

    highlight_name = None

    highlight_path = None


    print()
    print(
        "Generating AI Highlight..."
    )


    try:

        highlight_name = make_output_filename(
            filename,
            "highlight"
        )


        highlight_path = os.path.join(

            app.config["UPLOAD_FOLDER"],

            highlight_name

        )


        highlight_result = create_highlight(

            filepath,

            highlight_path

        )


        # create_highlight returns True/False

        if (
            highlight_result
            and
            os.path.exists(
                highlight_path
            )
        ):

            print(
                "✅ AI Highlight created:",
                highlight_path
            )

        else:

            print(
                "⚠️ AI Highlight was not created."
            )

            highlight_name = None

            highlight_path = None


    except Exception as e:

        print(
            "AI Highlight error:",
            e
        )

        traceback.print_exc()

        highlight_name = None

        highlight_path = None


    # ========================================================
    # ========================================================
    # AI ATTENTION MAP / GRAD-CAM
    # ========================================================
    # ========================================================

    gradcam_name = None

    gradcam_path = None


    print()
    print(
        "Generating AI Attention Map..."
    )


    try:

        if (
            AI_MODEL is not None
            and
            AI_TRANSFORM is not None
        ):


            # ------------------------------------------------
            # Load image for Grad-CAM
            # ------------------------------------------------

            from PIL import Image

            image = Image.open(
                filepath
            ).convert(
                "RGB"
            )


            # ------------------------------------------------
            # Apply same transform as prediction
            # ------------------------------------------------

            image_tensor = AI_TRANSFORM(
                image
            )


            image_tensor = (
                image_tensor
                .unsqueeze(0)
            )


            # ------------------------------------------------
            # Move tensor to same device as model
            # ------------------------------------------------

            try:

                device = next(
                    AI_MODEL.parameters()
                ).device

                image_tensor = (
                    image_tensor.to(
                        device
                    )
                )

            except Exception as e:

                print(
                    "Could not determine model device:",
                    e
                )


            # ------------------------------------------------
            # Output filename
            # ------------------------------------------------

            gradcam_name = make_output_filename(
                filename,
                "attention"
            )


            gradcam_path = os.path.join(

                app.config["UPLOAD_FOLDER"],

                gradcam_name

            )


            # ------------------------------------------------
            # Create Grad-CAM
            # ------------------------------------------------

            gradcam_result = create_gradcam(

                AI_MODEL,

                image_tensor,

                filepath,

                gradcam_path

            )


            if (
                gradcam_result
                and
                os.path.exists(
                    gradcam_path
                )
            ):

                print(
                    "✅ AI Attention Map created:",
                    gradcam_path
                )

            else:

                print(
                    "⚠️ AI Attention Map was not created."
                )

                gradcam_name = None

                gradcam_path = None


        else:

            print(
                "⚠️ AI model unavailable for Grad-CAM."
            )


    except Exception as e:

        print(
            "AI Attention Map error:",
            e
        )

        traceback.print_exc()

        gradcam_name = None

        gradcam_path = None


    # ========================================================
    # ========================================================
    # SEVERITY
    # ========================================================
    # ========================================================

    severity_level = "Unknown"

    affected_area = 0


    print()
    print(
        "Calculating severity..."
    )


    try:

        severity_result = estimate_severity(
            filepath
        )


        if isinstance(
            severity_result,
            dict
        ):

            severity_level = severity_result.get(
                "level",
                "Unknown"
            )

            affected_area = severity_result.get(
                "area",
                0
            )


        print(
            "Severity:",
            severity_level
        )

        print(
            "Affected area:",
            affected_area,
            "%"
        )


    except Exception as e:

        print(
            "Severity error:",
            e
        )

        traceback.print_exc()


        severity_level = "Unknown"

        affected_area = 0


    # ========================================================
    # ========================================================
    # SEVERITY ADVICE
    # ========================================================
    # ========================================================

    severity_advice = None


    print()
    print(
        "Generating severity advice..."
    )


    try:

        severity_advice = get_severity_advice(
            severity_level
        )


        if severity_advice is None:

            severity_advice = {

                "message":
                    "Unable to determine severity advice.",

                "actions": []

            }


        print(
            "Severity advice generated."
        )


    except Exception as e:

        print(
            "Severity advice error:",
            e
        )

        traceback.print_exc()


        severity_advice = {

            "message":
                "Unable to determine severity advice.",

            "actions": []

        }


    # ========================================================
    # ========================================================
    # AI EXPLANATION
    # ========================================================
    # ========================================================

    explanation = ""


    print()
    print(
        "Generating AI explanation..."
    )


    try:

        explanation = generate_explanation(

            info,

            confidence,

            severity_level,

            affected_area

        )


        print(
            "AI explanation generated."
        )


    except Exception as e:

        print(
            "AI explanation error:",
            e
        )

        traceback.print_exc()

        explanation = ""


    # ========================================================
    # ========================================================
    # PDF REPORT
    # ========================================================
    # ========================================================

    report_name = None

    report_path = None


    print()
    print(
        "Generating PDF report..."
    )


    try:

        report_name = make_output_filename(
            filename,
            "report"
        )


        # Change JPG extension to PDF

        report_name = os.path.splitext(
            report_name
        )[0] + ".pdf"


        report_path = os.path.join(

            app.config["UPLOAD_FOLDER"],

            report_name

        )


        # ----------------------------------------------------
        # create_report writes directly to filename.
        # It does not return a filename.
        # ----------------------------------------------------

        create_report(

            report_path,

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

            affected_area,

            info.get(
                "cause",
                "Information not available"
            ),

            info.get(
                "symptoms",
                []
            ),

            info.get(
                "treatment",
                []
            ),

            info.get(
                "organic_treatment",
                []
            ),

            info.get(
                "prevention",
                []
            )

        )


        if os.path.exists(
            report_path
        ):

            print(
                "✅ PDF report created:",
                report_path
            )

        else:

            print(
                "⚠️ PDF report file was not created."
            )

            report_name = None

            report_path = None


    except Exception as e:

        print(
            "PDF generation error:",
            e
        )

        traceback.print_exc()

        report_name = None

        report_path = None


    # ========================================================
    # ========================================================
    # SAVE HISTORY
    # ========================================================
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
            "✅ History saved."
        )


    except Exception as e:

        print(
            "History save error:",
            e
        )

        traceback.print_exc()


    # ========================================================
    # FINAL DEBUG INFORMATION
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL PLANTAI RESULT"
    )

    print(
        "=" * 70
    )

    print(
        "Original:",
        filename
    )

    print(
        "Highlight:",
        highlight_name
    )

    print(
        "Attention Map:",
        gradcam_name
    )

    print(
        "Prediction:",
        prediction
    )

    print(
        "Confidence:",
        confidence
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
        "Report:",
        report_name
    )

    print(
        "=" * 70
    )


    # ========================================================
    # RESULT PAGE
    # ========================================================

    return render_template(

        "result.html",

        # ----------------------------------------------------
        # Original image
        # ----------------------------------------------------

        image=filename,


        # ----------------------------------------------------
        # AI Highlight
        # ----------------------------------------------------

        highlight=highlight_name,


        # ----------------------------------------------------
        # AI Attention Map
        # ----------------------------------------------------

        gradcam=gradcam_name,


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction=prediction,

        confidence=round(
            confidence,
            2
        ),

        confidence_status=
            confidence_status,

        top_predictions=
            top_predictions,


        # ----------------------------------------------------
        # Severity
        # ----------------------------------------------------

        severity=severity_level,

        affected_area=affected_area,


        # ----------------------------------------------------
        # Disease information
        # ----------------------------------------------------

        info=info,


        # ----------------------------------------------------
        # AI explanation
        # ----------------------------------------------------

        explanation=explanation,


        # ----------------------------------------------------
        # Severity advice
        # ----------------------------------------------------

        severity_advice=
            severity_advice,


        # ----------------------------------------------------
        # PDF report
        # ----------------------------------------------------

        report=report_name

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
    print("=" * 70)
    print("🌱 PLANTAI MOBILE API REQUEST")
    print("=" * 70)


    # ========================================================
    # CHECK IMAGE
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


    filename = secure_filename(
        file.filename
    )


    if not filename:

        return jsonify({

            "success": False,

            "error":
                "Invalid file name"

        }), 400


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
            "Image save error:",
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

        traceback.print_exc()

        return jsonify({

            "success": False,

            "error":
                "AI prediction failed",

            "details":
                str(e)

        }), 500


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
    # SEVERITY
    # ========================================================

    severity_level = "Unknown"

    affected_area = 0


    try:

        severity_result = estimate_severity(
            filepath
        )


        if isinstance(
            severity_result,
            dict
        ):

            severity_level = severity_result.get(
                "level",
                "Unknown"
            )

            affected_area = severity_result.get(
                "area",
                0
            )

    except Exception as e:

        print(
            "API severity error:",
            e
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
    # EXPLANATION
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
            "API explanation error:",
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
            "API severity advice error:",
            e
        )

        severity_advice = {

            "message":
                "Unable to determine severity advice.",

            "actions": []

        }


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
            "API history error:",
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


    print(
        "Mobile API response prepared."
    )

    print(
        "=" * 70
    )


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
# SERVE GENERATED / UPLOADED FILES
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
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "ok",

        "service": "PlantAI"

    })


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