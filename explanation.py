def generate_explanation(
        info,
        confidence,
        severity,
        affected_area
):

    summary = f"""
🌱 AI Diagnosis Summary


Plant:
{info['plant']}


Disease:
{info['disease']}


Confidence:
{round(confidence,2)}%


Severity:
{severity}


Affected Area:
{round(affected_area,2)}%



Recommended Actions:

"""


    if severity == "Critical":

        summary += """
1. Remove severely infected leaves immediately.
2. Isolate affected plants.
3. Apply disease control treatment.
4. Monitor daily.
"""


    elif severity == "Moderate":

        summary += """
1. Remove infected portions.
2. Improve air circulation.
3. Apply recommended treatment.
4. Check plant regularly.
"""


    else:

        summary += """
1. Continue regular monitoring.
2. Maintain proper watering.
3. Follow preventive practices.
"""



    return summary