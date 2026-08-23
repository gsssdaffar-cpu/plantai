# -*- coding: utf-8 -*-

"""
============================================================
PlantAI Flask Application & Mobile REST API
============================================================

Features:
    - EfficientNet prediction through predict.py
    - 38 plant disease classes
    - AI Highlight ENABLED (Skipped for healthy plants)
    - AI Attention Map DISABLED
    - Lightweight OpenCV highlight
    - Severity estimation (Short-circuited for healthy plants)
    - Disease information & lookup
    - AI explanation & treatment advice
    - PDF report generation
    - Prediction history & Dashboard
    - Web Chatbot interface
    - JSON REST API for Mobile Apps (/api/v1/predict)
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
# FLASK APP & CONFIG
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

try:
    create_tables()
    print("Database tables initialized successfully.")
except Exception as e:
    print("Database initialization error:", e)


# ============================================================
# MODEL INITIALIZATION
# ============================================================

MODEL = None
TRANSFORM = None

try:
    MODEL = get_model()
    TRANSFORM = get_transform()
    print("PlantAI model initialized successfully.")
except Exception as e:
    print("Model initialization error:", e)
    traceback.print_exc()


# ============================================================
# UTILITIES
# ============================================================

def allowed_file(filename):
    if not filename or "." not in filename:
        return False
    extension = filename.rsplit(".", 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def safe_cleanup():
    try:
        cleanup()
    except Exception as e:
        print("Prediction cleanup warning:", e)
    try:
        gc.collect()
    except Exception:
        pass


def create_unique_filename(original_filename):
    safe_name = secure_filename(original_filename)
    if not safe_name:
        safe_name = "plant.jpg"
    unique_id = uuid.uuid4().hex[:12]
    return f"{unique_id}_{safe_name}"


# ============================================================
# WEB ROUTES
# ============================================================

@app.route("/")
def home():
    try:
        stats = dashboard_stats()
    except Exception as e:
        print("Dashboard stats error:", e)
        stats = {}

    try:
        recent = recent_predictions()
    except Exception as e:
        print("Recent predictions error:", e)
        recent = []

    return render_template("index.html", stats=stats, recent=recent)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "PlantAI API",
        "model_loaded": MODEL is not None,
        "ai_highlight": True,
        "severity": True,
        "pdf_report": True
    })


@app.route("/chatbot", methods=["GET", "POST"])
def chatbot():
    answer = ""
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            try:
                answer = chatbot_response(question)
            except Exception as e:
                print("Chatbot error:", e)
                answer = "Sorry, chatbot is currently unavailable."

    return render_template("chatbot.html", answer=answer)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serves uploaded images, generated highlights, and PDF reports."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/upload", methods=["GET"])
def upload_page():
    return render_template("upload.html")


# ============================================================
# PROCESSING HELPERS
# ============================================================

def fallback_disease_info(prediction):
    plant = "Unknown"
    disease = prediction

    if prediction and " - " in prediction:
        plant = prediction.split(" - ", 1)[0].strip()

    return {
        "plant": plant,
        "disease": disease,
        "cause": "Information not available",
        "symptoms": ["No disease information available"],
        "treatment": ["Consult appropriate plant disease guidance"],
        "organic_treatment": ["Maintain normal plant care"],
        "prevention": ["Monitor the plant regularly"]
    }


def calculate_severity(filepath, prediction=None):
    # Short-circuit logic for healthy leaves to prevent false positives from background/soil
    if prediction and "healthy" in prediction.lower():
        return "Healthy", 0.0

    severity_level = "Not calculated"
    affected_area = 0.0

    try:
        result = estimate_severity(filepath)
        if isinstance(result, dict):
            severity_level = result.get("level", result.get("severity", result.get("severity_level", "Not calculated")))
            affected_area = result.get("area", result.get("affected_area", result.get("affected", 0)))
        elif isinstance(result, tuple):
            if len(result) >= 2:
                severity_level, affected_area = result[0], result[1]
            elif len(result) == 1:
                severity_level = result[0]
        elif isinstance(result, str):
            severity_level = result
        elif isinstance(result, (int, float)):
            affected_area = float(result)

    except TypeError:
        try:
            result = estimate_severity(filepath, prediction)
            if isinstance(result, dict):
                severity_level = result.get("level", result.get("severity", "Not calculated"))
                affected_area = result.get("area", result.get("affected_area", 0))
            elif isinstance(result, tuple) and len(result) >= 2:
                severity_level, affected_area = result[0], result[1]
        except Exception as e:
            print("Severity fallback failed:", e)

    except Exception as e:
        print("Severity calculation failed:", e)
        traceback.print_exc()

    try:
        affected_area = float(affected_area)
    except Exception:
        affected_area = 0.0

    affected_area = max(0.0, min(100.0, affected_area))
    severity_level = str(severity_level) if severity_level else "Not calculated"

    return severity_level, round(affected_area, 2)


def generate_ai_explanation(prediction, confidence, severity, affected_area, info):
    """Robust wrapper trying multiple parameter combinations to fit explanation.py."""
    try:
        result = generate_explanation(prediction, confidence, severity, affected_area, info)
        if result is not None:
            return str(result)
    except Exception:
        pass

    try:
        result = generate_explanation(prediction, confidence, severity, affected_area)
        if result is not None:
            return str(result)
    except Exception:
        pass

    try:
        result = generate_explanation(prediction, confidence, info)
        if result is not None:
            return str(result)
    except Exception:
        pass

    return "Explanation currently unavailable."


def generate_severity_advice_safe(severity, prediction, affected_area):
    try:
        result = get_severity_advice(severity, affected_area, prediction)
        return result if result is not None else {}
    except TypeError:
        try:
            result = get_severity_advice(severity)
            return result if result is not None else {}
        except Exception as e:
            print("Severity advice fallback error:", e)
            return {}
    except Exception as e:
        print("Severity advice error:", e)
        return {}


def generate_highlight_safe(input_path, filename):
    try:
        print("Generating AI Highlight...")
        image = cv2.imread(input_path)
        if image is None:
            print("AI Highlight: image could not be read.")
            return None

        height, width = image.shape[:2]
        max_dimension = 900

        if max(height, width) > max_dimension:
            scale = max_dimension / float(max(height, width))
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_disease = np.array([8, 35, 30], dtype=np.uint8)
        upper_disease = np.array([45, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_disease, upper_disease)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(mask)
        minimum_area = max(20, int(image.shape[0] * image.shape[1] * 0.0001))

        for contour in contours:
            if cv2.contourArea(contour) >= minimum_area:
                cv2.drawContours(clean_mask, [contour], -1, 255, -1)

        mask = clean_mask
        overlay = image.copy()
        red_layer = np.zeros_like(image)
        red_layer[:, :, 2] = 255

        highlighted = cv2.addWeighted(overlay, 0.65, red_layer, 0.35, 0)
        result = image.copy()
        result[mask > 0] = highlighted[mask > 0]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) >= minimum_area:
                cv2.drawContours(result, [contour], -1, (0, 0, 255), 2)

        base_name = os.path.splitext(filename)[0]
        highlight_filename = f"{base_name}_highlight.jpg"
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], highlight_filename)

        success = cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            print("AI Highlight save failed.")
            return None

        return highlight_filename

    except Exception as e:
        print("AI Highlight generation failed:", e)
        traceback.print_exc()
        return None
    finally:
        gc.collect()


def generate_report_safe(filename, prediction, confidence, severity, affected_area, info, explanation, severity_advice):
    try:
        base_name = os.path.splitext(filename)[0]
        report_filename = f"{base_name}_report.pdf"

        plant = info.get("plant", "Unknown")
        disease = info.get("disease", prediction)
        cause = info.get("cause", "N/A")
        symptoms = info.get("symptoms", [])
        treatment = info.get("treatment", [])
        organic_treatment = info.get("organic_treatment", [])
        prevention = info.get("prevention", [])

        print("Creating PDF report...")
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
        if result and os.path.exists(result):
            return os.path.basename(result)

        target_path = os.path.join(app.config["UPLOAD_FOLDER"], report_filename)
        return report_filename if os.path.exists(target_path) else None

    except Exception as e:
        print("PDF generation failed:", e)
        traceback.print_exc()
        return None


def save_history_safe(filename, info, prediction, confidence, severity, affected_area):
    try:
        save_prediction(
            filename,
            info.get("plant", "Unknown"),
            info.get("disease", prediction),
            confidence,
            severity,
            affected_area
        )
        print("History saved.")
        return True
    except Exception as e:
        print("History save error:", e)
        return False


# ============================================================
# WEB UPLOAD ROUTE (HTML RESPONSE)
# ============================================================

@app.route("/upload", methods=["POST"])
def upload():
    print("\n" + "=" * 60)
    print("PLANTAI WEB UPLOAD")
    print("=" * 60)

    if "image" not in request.files:
        return "No image uploaded.", 400

    file = request.files["image"]
    if file.filename == "":
        return "No file selected.", 400

    if not allowed_file(file.filename):
        return "Invalid image format. Use JPG, JPEG, PNG or WEBP.", 400

    filename = create_unique_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(filepath)
        print("Image saved:", filepath)
    except Exception as e:
        print("File save error:", e)
        return "Could not save image.", 500

    try:
        print("Starting AI prediction...")
        prediction, confidence, top_predictions, confidence_status = predict_image(filepath)
        print("Prediction:", prediction)
        print(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        print("Prediction error:", e)
        traceback.print_exc()
        safe_cleanup()
        return f"Prediction failed: {str(e)}", 500
    finally:
        safe_cleanup()

    try:
        info = get_disease(prediction)
    except Exception as e:
        print("Disease lookup error:", e)
        info = None

    if info is None:
        print("Disease information not found.")
        info = fallback_disease_info(prediction)

    if info.get("plant", "Unknown") == "Unknown" and prediction and " - " in prediction:
        info["plant"] = prediction.split(" - ", 1)[0].strip()

    if confidence < 35.0:
        return render_template(
            "unknown.html",
            image=filename,
            confidence=round(confidence, 2),
            confidence_status=confidence_status,
            top_predictions=top_predictions
        )

    severity_level, affected_area = calculate_severity(filepath, prediction)
    explanation = generate_ai_explanation(
        prediction, confidence, severity_level, affected_area, info
    )
    severity_advice = generate_severity_advice_safe(
        severity_level, prediction, affected_area
    )

    if "healthy" in prediction.lower():
        highlight_file = None
    else:
        highlight_file = generate_highlight_safe(filepath, filename)

    report_file = generate_report_safe(
        filename, prediction, confidence, severity_level, affected_area, info, explanation, severity_advice
    )

    save_history_safe(
        filename, info, prediction, confidence, severity_level, affected_area
    )

    return render_template(
        "result.html",
        image=filename,
        highlight_image=highlight_file,
        report_file=report_file,
        prediction=prediction,
        confidence=round(confidence, 2),
        confidence_status=confidence_status,
        top_predictions=top_predictions,
        severity=severity_level,
        affected_area=affected_area,
        info=info,
        explanation=explanation,
        severity_advice=severity_advice
    )


# ============================================================
# MOBILE REST API ROUTE (JSON RESPONSE)
# ============================================================

@app.route("/api/v1/predict", methods=["POST"])
def api_predict():
    """
    PlantAI V2 Mobile REST API.

    Expects:
        multipart/form-data
        image = uploaded image

    Returns JSON compatible with Flutter DiagnosisModel.
    """

    filepath = None
    filename = None

    try:

        # ====================================================
        # 1. VALIDATE REQUEST
        # ====================================================

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image field found in request."
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file selected."
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": (
                    "Unsupported image format. "
                    "Use JPG, JPEG, PNG or WEBP."
                )
            }), 400

        # ====================================================
        # 2. SAVE TEMPORARY IMAGE
        # ====================================================

        filename = create_unique_filename(
            secure_filename(file.filename)
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        print()
        print("=" * 60)
        print("PLANTAI V2 MOBILE API")
        print("=" * 60)
        print("Image:", filename)

        # ====================================================
        # 3. AI PREDICTION
        # ====================================================

        prediction, confidence, top_predictions, confidence_status = (
            predict_image(filepath)
        )

        print("Prediction:", prediction)
        print(
            "Confidence:",
            f"{confidence:.2f}%"
        )

        # ====================================================
        # 4. DISEASE LOOKUP
        # ====================================================

        try:
            info = get_disease(prediction)
        except Exception as e:
            print("Disease lookup error:", e)
            info = None

        if info is None:
            info = fallback_disease_info(prediction)

        # ====================================================
        # 5. NORMALIZE DISEASE INFORMATION
        # ====================================================

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

        # ====================================================
        # 6. SEVERITY
        # ====================================================

        try:

            severity_level, affected_area = calculate_severity(
                filepath,
                prediction
            )

        except Exception as e:

            print(
                "Severity calculation error:",
                e
            )

            severity_level = "Unknown"
            affected_area = 0.0

        print(
            "Severity:",
            severity_level
        )

        print(
            "Affected area:",
            f"{affected_area:.2f}%"
        )

        # ====================================================
        # 7. EXPLANATION
        # ====================================================

        try:

            explanation = generate_ai_explanation(
                prediction,
                confidence,
                severity_level,
                affected_area,
                info
            )

        except Exception as e:

            print(
                "Explanation error:",
                e
            )

            explanation = ""

        # ====================================================
        # 8. SEVERITY ADVICE
        # ====================================================

        try:

            severity_advice = (
                generate_severity_advice_safe(
                    severity_level,
                    prediction,
                    affected_area
                )
            )

        except Exception as e:

            print(
                "Severity advice error:",
                e
            )

            severity_advice = []

        # ====================================================
        # 9. PREDICTION MARGIN
        # ====================================================

        prediction_margin = 100.0

        try:

            if (
                isinstance(top_predictions, list)
                and len(top_predictions) >= 2
            ):

                first = float(
                    top_predictions[0]["confidence"]
                )

                second = float(
                    top_predictions[1]["confidence"]
                )

                prediction_margin = (
                    first - second
                )

        except Exception as e:

            print(
                "Prediction margin error:",
                e
            )

        # ====================================================
        # 10. UNCERTAINTY
        # ====================================================

        low_confidence = (
            confidence < 70.0
        )

        ambiguous_prediction = (
            prediction_margin < 10.0
        )

        uncertain = (
            low_confidence
            or ambiguous_prediction
        )

        # ====================================================
        # 11. MOBILE RESPONSE
        # ====================================================

        response = {

            "success": True,

            # Flutter DiagnosisModel fields
            "plant": str(
                plant_name
            ),

            "disease": str(
                disease_name
            ),

            "confidence": (
                f"{confidence:.2f}%"
            ),

            "severity": str(
                severity_level
            ),

            "leaf_area_affected": (
                f"{affected_area:.2f}%"
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

            # Additional V2 fields
            "prediction": prediction,

            "confidence_status": (
                confidence_status
            ),

            "prediction_margin": (
                f"{prediction_margin:.2f}%"
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

            "explanation": (
                explanation
            ),

            "severity_advice": (
                severity_advice
            )
        }

        print()
        print(
            "Mobile API response ready."
        )

        print("=" * 60)

        return jsonify(response), 200

    except Exception as e:

        print()
        print("=" * 60)
        print("PLANTAI V2 MOBILE API ERROR")
        print("=" * 60)

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:

        # ====================================================
        # CLEANUP ONLY AFTER ALL PROCESSING IS COMPLETE
        # ====================================================

        if filepath:

            try:

                if os.path.exists(filepath):
                    os.remove(filepath)

                    print(
                        "Temporary image removed:",
                        filename
                    )

            except Exception as e:

                print(
                    "Cleanup error:",
                    e
                )

# ============================================================
# SERVER START
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))