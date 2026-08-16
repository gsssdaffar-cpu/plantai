from disease_lookup import DISEASE_DATABASE


def chatbot_response(question):

    question = question.lower().strip()


    # ========================================================
    # GENERAL QUESTIONS
    # ========================================================

    if not question:

        return (
            "Please ask me about a plant disease, "
            "symptoms, treatment, or prevention."
        )


    # ========================================================
    # SEARCH DISEASE DATABASE
    # ========================================================

    matches = []


    for key, info in DISEASE_DATABASE.items():

        searchable_text = " ".join([

            key,

            info.get(
                "plant",
                ""
            ),

            info.get(
                "disease",
                ""
            ),

            info.get(
                "cause",
                ""
            )

        ]).lower()


        if question in searchable_text:

            matches.append(
                info
            )


    # ========================================================
    # NO DIRECT MATCH
    # ========================================================

    if not matches:

        # Search individual words
        words = question.split()

        for key, info in DISEASE_DATABASE.items():

            searchable_text = " ".join([

                key,

                info.get(
                    "plant",
                    ""
                ),

                info.get(
                    "disease",
                    ""
                ),

                info.get(
                    "cause",
                    ""
                )

            ]).lower()


            if any(
                word in searchable_text
                for word in words
                if len(word) > 3
            ):

                matches.append(
                    info
                )


    # ========================================================
    # STILL NO MATCH
    # ========================================================

    if not matches:

        return (
            "I could not find that disease in my current "
            "PlantAI knowledge database. "
            "Please upload a clear leaf photograph for AI analysis."
        )


    # ========================================================
    # RETURN FIRST MATCH
    # ========================================================

    info = matches[0]


    response = []

    response.append(
        f"🌿 Plant: {info['plant']}"
    )

    response.append(
        f"🦠 Disease: {info['disease']}"
    )

    response.append(
        f"🧬 Cause: {info['cause']}"
    )


    symptoms = info.get(
        "symptoms",
        []
    )

    if symptoms:

        response.append(
            "\n🔍 Symptoms:"
        )

        for symptom in symptoms:

            response.append(
                f"• {symptom}"
            )


    treatment = info.get(
        "treatment",
        []
    )

    if treatment:

        response.append(
            "\n💊 Treatment:"
        )

        for item in treatment:

            response.append(
                f"• {item}"
            )


    prevention = info.get(
        "prevention",
        []
    )

    if prevention:

        response.append(
            "\n🛡 Prevention:"
        )

        for item in prevention:

            response.append(
                f"• {item}"
            )


    return "\n".join(
        response
    )