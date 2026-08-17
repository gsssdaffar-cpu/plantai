# -*- coding: utf-8 -*-

import os
import inspect
import traceback

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

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# MAX UPLOAD SIZE
# ============================================================

# 10 MB maximum image upload

app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

try:

    create_tables()

    print()
    print("=" * 60)
    print("Database tables initialized.")
    print("=" * 60)

except Exception as e:

    print()
    print("=" * 60)
    print("Database initialization error")
    print("=" * 60)

    print(e)

    traceback.print_exc()


# ============================================================
# HELPER
# ============================================================

def safe_call(
    function,
    possible_values=None,
    default=None,
    function_name="function"
):

    """
    Safely call a project function.

    This helper allows PlantAI to work even if the helper
    functions have slightly different parameter names.

    It examines the function signature and supplies the
    parameters that are available.
    """

    if possible_values is None:

        possible_values = {}


    try:

        signature = inspect.signature(
            function
        )

        parameters = signature.parameters


        kwargs = {}


        # ----------------------------------------------------
        # Match parameters by name
        # ----------------------------------------------------

        for parameter_name in parameters:

            parameter = parameters[
                parameter_name
            ]


            # Ignore *args / **kwargs

            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD
            ):

                continue


            # ------------------------------------------------
            # Exact parameter match
            # ------------------------------------------------

            if parameter_name in possible_values:

                kwargs[
                    parameter_name
                ] = possible_values[
                    parameter_name
                ]

                continue


            # ------------------------------------------------
            # Common aliases
            # ------------------------------------------------

            aliases = {

                "image":
                    [
                        "image_path",
                        "filepath",
                        "file_path",
                        "path",
                        "image_file"
                    ],

                "image_path":
                    [
                        "image",
                        "filepath",
                        "file_path",
                        "path",
                        "image_file"
                    ],

                "filepath":
                    [
                        "image_path",
                        "image",
                        "file_path",
                        "path"
                    ],

                "file_path":
                    [
                        "filepath",
                        "image_path",
                        "image",
                        "path"
                    ],

                "prediction":
                    [
                        "disease",
                        "class_name",
                        "label"
                    ],

                "disease":
                    [
                        "prediction",
                        "class_name",
                        "label"
                    ],

                "confidence":
                    [
                        "score",
                        "probability"
                    ],

                "info":
                    [
                        "disease_info",
                        "disease_data"
                    ],

                "severity":
                    [
                        "severity_level"
                    ],

                "affected_area":
                    [
                        "area",
                        "percentage",
                        "affected_percentage"
                    ],

                "plant":
                    [
                        "plant_name"
                    ],

                "filename":
                    [
                        "file_name"
                    ],

                "explanation":
                    [
                        "ai_explanation"
                    ],

                "severity_advice":
                    [
                        "advice"
                    ]

            }


            found = False


            # ------------------------------------------------
            # Search aliases
            # ------------------------------------------------

            if parameter_name in aliases:

                for alias in aliases[
                    parameter_name
                ]:

                    if alias in possible_values:

                        kwargs[
                            parameter_name
                        ] = possible_values[
                            alias
                        ]

                        found = True

                        break


            if found:

                continue


            # ------------------------------------------------
            # Parameter has default
            # ------------------------------------------------

            if (
                parameter.default
                is not inspect.Parameter.empty
            ):

                continue


        # ----------------------------------------------------
        # Call function
        # ----------------------------------------------------

        return function(
            **kwargs
        )


    except Exception as e:

        print()
        print(
            f"{function_name} error:"
        )

        print(e)

        traceback.print_exc()

        return default


# ============================================================
# NORMALIZE FEATURE RESULT
# ============================================================

def normalize_filename(
    value
):

    """
    Convert a returned file/path value into a filename
    usable by /uploads/<filename>.
    """

    if value is None:

        return None


    if isinstance(
        value,
        dict
    ):

        for key in [
            "filename",
            "file",
            "path",
            "file_path",
            "image",
            "output"
        ]:

            if key in value:

                value = value[key]

                break


    if not isinstance(
        value,
        str
    ):

        return None


    value = value.strip()


    if not value:

        return None


    # --------------------------------------------------------
    # Convert path to basename
    # --------------------------------------------------------

    return os.path.basename(
        value
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
# MAIN AI UPLOAD
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
    # CHECK FILE
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


    if file.filename == "":

        print(
            "ERROR: No file selected."
        )

        return (
            "No file selected.",
            400
        )


    # ========================================================
    # SECURE FILENAME
    # ========================================================

    filename = secure_filename(
        file.filename
    )


    if not filename:

        return (
            "Invalid file name.",
            400
        )


    # ========================================================
    # CREATE UNIQUE FILENAME
    # ========================================================

    import uuid


    extension = os.path.splitext(
        filename
    )[1].lower()


    unique_filename = (
        uuid.uuid4().hex
        + extension
    )


    filepath = os.path.join(

        app.config[
            "UPLOAD_FOLDER"
        ],

        unique_filename

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

        traceback.print_exc()

        return (
            "Could not save image.",
            500
        )


    # ========================================================
    # AI PREDICTION
    # ========================================================

    print()
    print("-" * 70)
    print("STEP 1: AI PREDICTION")
    print("-" * 70)


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


        return (

            f"Prediction failed: {str(e)}",

            500

        )


    # ========================================================
    # DISEASE DATABASE
    # ========================================================

    print()
    print("-" * 70)
    print("STEP 2: DISEASE DATABASE")
    print("-" * 70)


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
    # FALLBACK
    # ========================================================

    if info is None:

        print(
            "WARNING: Disease not found:"
        )

        print(
            prediction
        )


        info = {

            "plant":
                "Unknown",

            "disease":
                prediction,

            "cause":
                "Information not available",

            "symptoms":
                [],

            "treatment":
                [],

            "organic_treatment":
                [],

            "prevention":
                []

        }


    # ========================================================
    # ENSURE DISEASE NAME
    # ========================================================

    disease_name = info.get(
        "disease",
        prediction
    )


    if not disease_name:

        disease_name = prediction


    print(
        "Plant:",
        info.get(
            "plant",
            "Unknown"
        )
    )


    print(
        "Disease:",
        disease_name
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

            image=unique_filename,

            confidence=round(
                confidence,
                2
            ),

            prediction=prediction,

            top_predictions=top_predictions,

            confidence_status=
                confidence_status

        )


    # ========================================================
    # STEP 3 - AI HIGHLIGHT
    # ========================================================

    print()
    print("-" * 70)
    print("STEP 3: AI HIGHLIGHT")
    print("-" * 70)


    highlight_name = None


    try:

        highlight_result = safe_call(

            create_highlight,

            {

                "image_path":
                    filepath,

                "filepath":
                    filepath,

                "image":
                    filepath,

                "prediction":
                    prediction,

                "disease":
                    disease_name

            },

            default=None,

            function_name=
                "AI Highlight"

        )


        highlight_name = normalize_filename(
            highlight_result
        )


        print(
            "Highlight result:",
            highlight_result
        )


        print(
            "Highlight filename:",
            highlight_name
        )


    except Exception as e:

        print(
            "Highlight failed:",
            e
        )

        traceback.print_exc()


    # ========================================================
    # STEP 4 - AI ATTENTION MAP / GRAD-CAM
    # ========================================================

    print()
    print("-" * 70)
    print("STEP 4: AI ATTENTION MAP")
    print("-" * 70)


    gradcam_name = None


    # --------------------------------------------------------
    # Try importing Grad-CAM module
    # --------------------------------------------------------

    try:

        from gradcam import create_gradcam


        gradcam_result = safe_call(

            create_gradcam,

            {

                "image_path":
                    filepath,

                "filepath":
                    filepath,

                "image":
                    filepath,

                "prediction":
                    prediction,

                "disease":
                    disease_name,

                "model":
                    get_model(),

                "transform":
                    get_transform(),

                "predicted_class":
                    prediction

            },

            default=None,

            function_name=
                "Grad-CAM"

        )


        gradcam_name = normalize_filename(
            gradcam_result
        )


        print(
            "Grad-CAM result:",
            gradcam_result
        )


        print(
            "Grad-CAM filename:",
            gradcam_name
        )


    except ImportError:

        print(
            "Grad-CAM module not available."
        )


    except Exception as e:

        print(
            "Grad-CAM failed:",
            e
        )

        traceback.print_exc()


    # ========================================================
    # STEP 5 - SEVERITY
    # ========================================================

    print()
    print("-" * 70)
    print("STEP 5: SEVERITY ANALYSIS")
    print("-" * 70)


    severity_level = (
        "Not calculated"
    )

    affected_area = 0


    try:

        severity_result = safe_call(

            estimate_severity,

            {

                "image_path":
                    filepath,

                "filepath":
                    filepath,

                "image":
                    filepath,

                "prediction":
                    prediction,

                "disease":
                    disease_name

            },

            default=None,

            function_name=
                "Severity analysis"

        )


        print(
            "Severity raw result:",
            severity_result
        )


        # ----------------------------------------------------
        # Dictionary result
        # ----------------------------------------------------

        if isinstance(
            severity_result,
            dict
        ):

            severity_level = (
                severity_result.get(
                    "severity",
                    severity_result.get(
                        "level",
                        "Not calculated"
                    )
                )
            )


            affected_area = (
                severity_result.get(
                    "affected_area",
                    severity_result.get(
                        "area",
                        severity_result.get(
                            "percentage",
                            0
                        )
                    )
                )
            )


        # ----------------------------------------------------
        # Tuple/list result
        # ----------------------------------------------------

        elif isinstance(
            severity_result,
            (tuple, list)
        ):

            if len(
                severity_result
            ) >= 1:

                severity_level = (
                    severity_result[0]
                )


            if len(
                severity_result
            ) >= 2:

                affected_area = (
                    severity_result[1]
                )


        # ----------------------------------------------------
        # String result
        # ----------------------------------------------------

        elif isinstance(
            severity_result,
            str
        ):

            severity_level = (
                severity_result
            )


    except Exception as e:

        print(
            "Severity analysis failed:",
            e
        )

        traceback.print_exc()


    # ========================================================
    # NORMALIZE AFFECTED AREA
    # ========================================================

    try:

        affected_area = float(
            affected_area
        )

        affected_area = round(
            affected_area,
            2
        )

    except Exception:

        affected_area = 0


    print(
        "Severity:",
        severity_level
    )

    print(
        "Affected area:",
        affected_area
    )


    # ========================================================
    # STEP 6 - AI EXPLANATION
    # ========================================================

    print()
    print("-" * 70)
    print("STEP 6: AI EXPLANATION")
    print("-" * 70)


    explanation = ""


    try:

        explanation_result = safe_call(

            generate_explanation,

            {

                "prediction":
                    prediction,

                "disease":
                    disease_name,

                "confidence":
                    confidence,

                "info":
                    info,

                "plant":
                    info.get(
                        "plant",
                        "Unknown"
                    ),

                "severity":
                    severity_level,

                "affected_area":
                    affected_area,

                "top_predictions":
                    top_predictions

            },

            default="",

            function_name=
                "AI explanation"

        )


        if explanation_result is not None:

            if isinstance(
                explanation_result,
                dict
            ):

                explanation = (
                    explanation_result.get(
                        "explanation",
                        explanation_result.get(
                            "text",
                            ""
                        )
                    )
                )

            else:

                explanation = str(
                    explanation_result
                )


        print(
            "Explanation:",
            explanation
        )


    except Exception as e:

        print(
            "Explanation failed:",
            e
        )

        traceback.print_exc()


    # ========================================================
    # STEP 7 - SEVERITY ADVICE
    # ========================================================

    print()
    print("-" * 70)
    print("STEP 7: SEVERITY ADVICE")
    print("-" * 70)


    severity_advice = ""


    try:

        advice_result = safe_call(

            get_severity_advice,

            {

                "severity":
                    severity_level,

                "severity_level":
                    severity_level,

                "affected_area":
                    affected_area,

                "area":
                    affected_area,

                "prediction":
                    prediction,

                "disease":
                    disease_name,

                "info":
                    info

            },

            default="",

            function_name=
                "Severity advice"

        )


        if advice_result is not None:

            if isinstance(
                advice_result,
                dict
            ):

                severity_advice = (
                    advice_result.get(
                        "advice",
                        advice_result.get(
                            "text",
                            ""
                        )
                    )
                )

            else:

                severity_advice = str(
                    advice_result
                )


        print(
            "Severity advice:",
            severity_advice
        )


    except Exception as e:

        print(
            "Severity advice failed:",
            e
        )

        traceback.print_exc()


    # ========================================================
    # STEP 8 - PDF REPORT
    # ========================================================

    print()
    print("-" * 70)
    print("STEP 8: PDF REPORT")
    print("-" * 70)


    report_name = None


    try:

        report_result = safe_call(

            create_report,

            {

                "image_path":
                    filepath,

                "filepath":
                    filepath,

                "image":
                    filepath,

                "filename":
                    unique_filename,

                "prediction":
                    prediction,

                "disease":
                    disease_name,

                "confidence":
                    confidence,

                "confidence_status":
                    confidence_status,

                "top_predictions":
                    top_predictions,

                "plant":
                    info.get(
                        "plant",
                        "Unknown"
                    ),

                "info":
                    info,

                "severity":
                    severity_level,

                "severity_level":
                    severity_level,

                "affected_area":
                    affected_area,

                "explanation":
                    explanation,

                "severity_advice":
                    severity_advice,

                "highlight":
                    highlight_name,

                "gradcam":
                    gradcam_name

            },

            default=None,

            function_name=
                "PDF report"

        )


        report_name = normalize_filename(
            report_result
        )


        print(
            "PDF result:",
            report_result
        )


        print(
            "PDF filename:",
            report_name
        )


    except Exception as e:

        print(
            "PDF generation failed:",
            e
        )

        traceback.print_exc()


    # ========================================================
    # STEP 9 - SAVE HISTORY
    # ========================================================

    print()
    print("-" * 70)
    print("STEP 9: SAVE HISTORY")
    print("-" * 70)


    try:

        save_prediction(

            unique_filename,

            info.get(
                "plant",
                "Unknown"
            ),

            disease_name,

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

        traceback.print_exc()


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("🌱 PLANTAI RESULT")
    print("=" * 70)

    print(
        "Image:",
        unique_filename
    )

    print(
        "Plant:",
        info.get(
            "plant",
            "Unknown"
        )
    )

    print(
        "Disease:",
        disease_name
    )

    print(
        f"Confidence: "
        f"{confidence:.2f}%"
    )

    print(
        "Severity:",
        severity_level
    )

    print(
        "Affected area:",
        affected_area
    )

    print(
        "Highlight:",
        highlight_name
    )

    print(
        "Attention map:",
        gradcam_name
    )

    print(
        "PDF:",
        report_name
    )

    print("=" * 70)


    # ========================================================
    # RESULT TEMPLATE
    # ========================================================

    return render_template(

        "result.html",

        # ----------------------------------------------------
        # Original image
        # ----------------------------------------------------

        image=unique_filename,


        # ----------------------------------------------------
        # AI highlight
        # ----------------------------------------------------

        highlight=highlight_name,


        # ----------------------------------------------------
        # AI attention map
        # ----------------------------------------------------

        gradcam=gradcam_name,


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction=prediction,

        plant=info.get(
            "plant",
            "Unknown"
        ),

        disease=disease_name,


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence=round(
            confidence,
            2
        ),

        confidence_status=
            confidence_status,


        # ----------------------------------------------------
        # Top predictions
        # ----------------------------------------------------

        top_predictions=
            top_predictions,


        # ----------------------------------------------------
        # Severity
        # ----------------------------------------------------

        severity=
            severity_level,

        affected_area=
            affected_area,


        # ----------------------------------------------------
        # Disease information
        # ----------------------------------------------------

        info=info,


        # ----------------------------------------------------
        # Individual fields
        #
        # These are useful if result.html uses direct
        # variables instead of info["..."].
        # ----------------------------------------------------

        cause=info.get(
            "cause",
            ""
        ),

        symptoms=info.get(
            "symptoms",
            []
        ),

        treatment=info.get(
            "treatment",
            []
        ),

        organic_treatment=info.get(
            "organic_treatment",
            []
        ),

        prevention=info.get(
            "prevention",
            []
        ),


        # ----------------------------------------------------
        # AI explanation
        # ----------------------------------------------------

        explanation=
            explanation,


        # ----------------------------------------------------
        # Severity advice
        # ----------------------------------------------------

        severity_advice=
            severity_advice,


        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------

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
    print("=" * 70)
    print("🌱 PLANTAI MOBILE API")
    print("=" * 70)


    # ========================================================
    # CHECK FILE
    # ========================================================

    if "image" not in request.files:

        return {

            "success":
                False,

            "error":
                "No image uploaded"

        }, 400


    file = request.files[
        "image"
    ]


    if file.filename == "":

        return {

            "success":
                False,

            "error":
                "No file selected"

        }, 400


    # ========================================================
    # UNIQUE FILE
    # ========================================================

    filename = secure_filename(
        file.filename
    )


    if not filename:

        return {

            "success":
                False,

            "error":
                "Invalid file name"

        }, 400


    import uuid


    extension = os.path.splitext(
        filename
    )[1].lower()


    unique_filename = (
        uuid.uuid4().hex
        + extension
    )


    filepath = os.path.join(

        app.config[
            "UPLOAD_FOLDER"
        ],

        unique_filename

    )


    # ========================================================
    # SAVE
    # ========================================================

    try:

        file.save(
            filepath
        )

    except Exception as e:

        return {

            "success":
                False,

            "error":
                "Could not save image",

            "details":
                str(e)

        }, 500


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


        return {

            "success":
                False,

            "error":
                "AI prediction failed",

            "details":
                str(e)

        }, 500


    # ========================================================
    # DATABASE
    # ========================================================

    try:

        info = get_disease(
            prediction
        )

    except Exception:

        info = None


    if info is None:

        info = {

            "plant":
                "Unknown",

            "disease":
                prediction,

            "cause":
                "Information not available",

            "symptoms":
                [],

            "treatment":
                [],

            "organic_treatment":
                [],

            "prevention":
                []

        }


    disease_name = info.get(
        "disease",
        prediction
    )


    # ========================================================
    # LOW CONFIDENCE
    # ========================================================

    if confidence < 40:

        return {

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
                disease_name,

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
    # OPTIONAL AI FEATURES
    # ========================================================

    severity_level = (
        "Not calculated"
    )

    affected_area = 0


    try:

        severity_result = safe_call(

            estimate_severity,

            {

                "image_path":
                    filepath,

                "filepath":
                    filepath,

                "image":
                    filepath,

                "prediction":
                    prediction,

                "disease":
                    disease_name

            },

            default=None,

            function_name=
                "API severity"

        )


        if isinstance(
            severity_result,
            dict
        ):

            severity_level = (
                severity_result.get(
                    "severity",
                    severity_result.get(
                        "level",
                        "Not calculated"
                    )
                )
            )

            affected_area = (
                severity_result.get(
                    "affected_area",
                    severity_result.get(
                        "area",
                        0
                    )
                )
            )

        elif isinstance(
            severity_result,
            (tuple, list)
        ):

            if len(
                severity_result
            ) >= 1:

                severity_level = (
                    severity_result[0]
                )

            if len(
                severity_result
            ) >= 2:

                affected_area = (
                    severity_result[1]
                )

    except Exception:

        pass


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    try:

        save_prediction(

            unique_filename,

            info.get(
                "plant",
                "Unknown"
            ),

            disease_name,

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

    return {

        "success":
            True,

        "recognized":
            True,

        "image":
            unique_filename,

        "prediction":
            prediction,

        "plant":
            info.get(
                "plant",
                "Unknown"
            ),

        "disease":
            disease_name,

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

        "top_predictions":
            top_predictions

    }


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
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {

        "status":
            "ok",

        "service":
            "PlantAI"

    }


# ============================================================
# RUN LOCALLY
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