from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

from predict import predict_image

from disease_lookup import get_disease

from severity import estimate_severity

from severity_advice import get_severity_advice


# ============================================================
# FLASK API
# ============================================================

app = Flask(__name__)


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
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "success": True,
        "message": "PlantAI API is running"
    })


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({

            "success": False,

            "error": "No image uploaded"

        }), 400


    file = request.files["image"]


    if file.filename == "":

        return jsonify({

            "success": False,

            "error": "No file selected"

        }), 400


    # --------------------------------------------------------
    # Secure filename
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )


    if not filename:

        return jsonify({

            "success": False,

            "error": "Invalid filename"

        }), 400


    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    filepath = os.path.join(

        app.config["UPLOAD_FOLDER"],

        filename

    )


    file.save(
        filepath
    )


    try:

        # ====================================================
        # AI PREDICTION
        # ====================================================

        prediction, confidence, top_predictions, confidence_status = (
            predict_image(
                filepath
            )
        )


        # ====================================================
        # DISEASE INFORMATION
        # ====================================================

        info = get_disease(
            prediction
        )


        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        if info is None:

            info = {

                "plant": "Unknown",

                "disease": prediction,

                "cause": "Information not available",

                "symptoms": [],

                "treatment": [],

                "organic_treatment": [],

                "prevention": []

            }


        # ====================================================
        # SEVERITY ANALYSIS
        # ====================================================

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


        # ====================================================
        # SEVERITY ADVICE
        # ====================================================

        severity_advice = get_severity_advice(
            severity_level
        )


        # ====================================================
        # RETURN MOBILE JSON
        # ====================================================

        return jsonify({

            "success": True,

            "image": filename,

            "plant": info.get(
                "plant",
                "Unknown"
            ),

            "disease": info.get(
                "disease",
                prediction
            ),

            "prediction": prediction,

            "confidence": round(
                confidence,
                2
            ),

            "confidence_status": confidence_status,

            "severity": severity_level,

            "affected_area": affected_area,

            "cause": info.get(
                "cause",
                "Information not available"
            ),

            "symptoms": info.get(
                "symptoms",
                []
            ),

            "treatment": info.get(
                "treatment",
                []
            ),

            "organic_treatment": info.get(
                "organic_treatment",
                []
            ),

            "prevention": info.get(
                "prevention",
                []
            ),

            "severity_advice": severity_advice,

            "top_predictions": top_predictions

        })


    except Exception as e:

        print()
        print("=" * 60)
        print("❌ API PREDICTION ERROR")
        print("=" * 60)
        print(e)
        print("=" * 60)


        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


# ============================================================
# RUN API
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("🌱 PLANTAI MOBILE API")
    print("=" * 60)
    print()
    print("API running at:")
    print("http://127.0.0.1:5001")
    print()
    print("Prediction endpoint:")
    print("POST /api/predict")
    print()
    print("=" * 60)


    app.run(

        host="0.0.0.0",

        port=5001,

        debug=True

    )