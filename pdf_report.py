import os
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

REPORT_FOLDER = "reports"
os.makedirs(REPORT_FOLDER, exist_ok=True)


def generate_report(info, confidence):

    filename = os.path.join(REPORT_FOLDER, "plant_report.pdf")

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("<b>PlantAI Disease Report</b>", styles["Title"]))
    story.append(Paragraph(f"<b>Plant:</b> {info['plant']}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Disease:</b> {info['disease']}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles["BodyText"]))
    story.append(Paragraph(f"<b>Cause:</b> {info['cause']}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Symptoms:</b> {info['symptoms']}", styles["BodyText"]))

    story.append(Paragraph("<b>Treatment</b>", styles["Heading2"]))
    for item in info["treatment"]:
        story.append(Paragraph("• " + item.strip(), styles["BodyText"]))

    story.append(Paragraph("<b>Prevention</b>", styles["Heading2"]))
    for item in info["prevention"]:
        story.append(Paragraph("• " + item.strip(), styles["BodyText"]))

    doc.build(story)

    return filename