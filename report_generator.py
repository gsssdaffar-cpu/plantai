from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet



def create_report(
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
):


    pdf = SimpleDocTemplate(
        filename
    )


    styles = getSampleStyleSheet()

    content = []



    content.append(
        Paragraph(
            "PlantAI Disease Diagnosis Report",
            styles["Title"]
        )
    )


    content.append(
        Spacer(1,20)
    )



    data = [

        f"Plant: {plant}",

        f"Disease: {disease}",

        f"Confidence: {confidence} %",

        f"Severity: {severity}",

        f"Leaf Area Affected: {affected_area} %",

        "",

        f"Cause: {cause}",

        "",

        "Symptoms:",

    ]



    for item in data:

        content.append(
            Paragraph(
                item,
                styles["Normal"]
            )
        )

        content.append(
            Spacer(1,8)
        )



    for s in symptoms:

        content.append(
            Paragraph(
                "• " + s,
                styles["Normal"]
            )
        )



    content.append(
        Spacer(1,15)
    )


    content.append(
        Paragraph(
            "Treatment:",
            styles["Heading2"]
        )
    )


    for t in treatment:

        content.append(
            Paragraph(
                "• " + t,
                styles["Normal"]
            )
        )



    content.append(
        Spacer(1,15)
    )



    content.append(
        Paragraph(
            "Organic Treatment:",
            styles["Heading2"]
        )
    )


    for o in organic_treatment:

        content.append(
            Paragraph(
                "• " + o,
                styles["Normal"]
            )
        )



    content.append(
        Spacer(1,15)
    )



    content.append(
        Paragraph(
            "Prevention:",
            styles["Heading2"]
        )
    )


    for p in prevention:

        content.append(
            Paragraph(
                "• " + p,
                styles["Normal"]
            )
        )



    pdf.build(content)