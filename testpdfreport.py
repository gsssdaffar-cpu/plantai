from pdf_report import generate_report

info = {
    "plant": "Tomato",
    "disease": "Target Spot",
    "cause": "Corynespora cassiicola fungus",
    "symptoms": "Brown circular spots",
    "treatment": ["Apply fungicide", "Remove infected leaves"],
    "prevention": ["Improve air circulation"]
}

generate_report(info, 99.97)

print("PDF Generated")