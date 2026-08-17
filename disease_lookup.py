# ============================================================
# LOOKUP FUNCTION
# ============================================================

def get_disease(disease_name):

    if not disease_name:
        print("WARNING: Empty disease name")
        return None

    original_name = str(disease_name).strip()

    print()
    print("=" * 60)
    print("DISEASE LOOKUP")
    print("=" * 60)

    print(
        "Model prediction:",
        repr(original_name)
    )


    # ========================================================
    # NORMALIZE
    # ========================================================

    def normalize(name):

        name = str(name).strip()

        # Lowercase
        name = name.lower()

        # Convert common separators
        name = name.replace("___", "|||")
        name = name.replace("_", " ")

        # Restore separator
        name = name.replace("|||", "___")

        # Remove duplicate spaces
        name = " ".join(name.split())

        # Normalize brackets
        name = name.replace("(", " ")
        name = name.replace(")", " ")

        # Normalize punctuation
        name = name.replace(",", " ")

        # Corn normalization
        name = name.replace(
            "corn maize",
            "corn"
        )

        name = name.replace(
            "corn_(maize)",
            "corn"
        )

        # Bell pepper normalization
        name = name.replace(
            "pepper bell",
            "pepper bell"
        )

        # Remove trailing underscore
        name = name.rstrip("_")

        return name.strip()


    normalized_prediction = normalize(
        original_name
    )

    print(
        "Normalized prediction:",
        repr(normalized_prediction)
    )


    # ========================================================
    # 1. EXACT DATABASE KEY MATCH
    # ========================================================

    for key, value in DISEASE_DATABASE.items():

        if normalize(key) == normalized_prediction:

            print(
                "EXACT MATCH:",
                key
            )

            print("=" * 60)

            return value


    # ========================================================
    # 2. HANDLE FORMATTED PREDICTION
    #
    # Example:
    #
    # Corn (maize) - Cercospora leaf spot Gray leaf spot
    #
    # Database:
    #
    # Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
    # ========================================================

    formatted_prediction = (
        original_name
        .replace(" - ", "___")
    )

    formatted_normalized = normalize(
        formatted_prediction
    )

    print(
        "Formatted normalized:",
        repr(formatted_normalized)
    )


    for key, value in DISEASE_DATABASE.items():

        key_normalized = normalize(key)

        if key_normalized == formatted_normalized:

            print(
                "FORMATTED MATCH:",
                key
            )

            print("=" * 60)

            return value


    # ========================================================
    # 3. PLANT + DISEASE MATCH
    # ========================================================

    if " - " in original_name:

        prediction_plant, prediction_disease = (
            original_name.split(
                " - ",
                1
            )
        )

        prediction_plant = normalize(
            prediction_plant
        )

        prediction_disease = normalize(
            prediction_disease
        )

        print(
            "Prediction plant:",
            prediction_plant
        )

        print(
            "Prediction disease:",
            prediction_disease
        )


        for key, value in DISEASE_DATABASE.items():

            if "___" not in key:
                continue

            database_plant, database_disease = (
                key.split(
                    "___",
                    1
                )
            )

            database_plant = normalize(
                database_plant
            )

            database_disease = normalize(
                database_disease
            )


            # ------------------------------------------------
            # Plant comparison
            # ------------------------------------------------

            plant_match = (

                prediction_plant
                ==
                database_plant

            )


            # ------------------------------------------------
            # Disease comparison
            # ------------------------------------------------

            disease_match = (

                prediction_disease
                ==
                database_disease

            )


            if plant_match and disease_match:

                print(
                    "PLANT + DISEASE MATCH:",
                    key
                )

                print("=" * 60)

                return value


    # ========================================================
    # 4. SPECIAL CORN MATCH
    # ========================================================

    prediction_lower = original_name.lower()


    # Cercospora / Gray Leaf Spot

    if (

        "corn" in prediction_lower

        and

        (
            "cercospora" in prediction_lower
            or
            "gray leaf spot" in prediction_lower
        )

    ):

        key = (
            "Corn_(maize)___"
            "Cercospora_leaf_spot Gray_leaf_spot"
        )


        if key in DISEASE_DATABASE:

            print(
                "SPECIAL CORN CERCOSPORA MATCH"
            )

            print(
                "Matched:",
                key
            )

            print("=" * 60)

            return DISEASE_DATABASE[key]


    # ========================================================
    # 5. CORN NORTHERN LEAF BLIGHT
    # ========================================================

    if (

        "corn" in prediction_lower

        and

        "northern leaf blight"
        in prediction_lower

    ):

        key = (
            "Corn_(maize)___"
            "Northern_Leaf_Blight"
        )


        if key in DISEASE_DATABASE:

            print(
                "SPECIAL CORN NORTHERN BLIGHT MATCH"
            )

            print("=" * 60)

            return DISEASE_DATABASE[key]


    # ========================================================
    # 6. CORN COMMON RUST
    # ========================================================

    if (

        "corn" in prediction_lower

        and

        "common rust"
        in prediction_lower

    ):

        key = (
            "Corn_(maize)___"
            "Common_rust_"
        )


        if key in DISEASE_DATABASE:

            print(
                "SPECIAL CORN COMMON RUST MATCH"
            )

            print("=" * 60)

            return DISEASE_DATABASE[key]


    # ========================================================
    # 7. PARTIAL MATCH
    # ========================================================

    prediction_words = set(
        normalized_prediction
        .replace("___", " ")
        .split()
    )


    best_match = None

    best_score = 0


    for key, value in DISEASE_DATABASE.items():

        key_normalized = normalize(key)

        key_words = set(
            key_normalized
            .replace("___", " ")
            .split()
        )


        if not prediction_words:
            continue


        common_words = (
            prediction_words
            &
            key_words
        )


        score = len(common_words)


        if score > best_score:

            best_score = score

            best_match = value


            best_key = key


    # Require a reasonable match

    if (

        best_match is not None

        and

        best_score >= 2

    ):

        print(
            "BEST PARTIAL MATCH:",
            best_key
        )

        print(
            "Match score:",
            best_score
        )

        print("=" * 60)

        return best_match


    # ========================================================
    # 8. NO MATCH
    # ========================================================

    print(
        "WARNING: Disease not found in database:"
    )

    print(
        repr(original_name)
    )

    print(
        "Available database entries:",
        len(DISEASE_DATABASE)
    )

    print("=" * 60)

    return None