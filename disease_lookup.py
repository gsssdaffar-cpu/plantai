# -*- coding: utf-8 -*-

# ============================================================
# PLANTAI DISEASE DATABASE
# 38 PLANTVILLAGE CLASSES
# ============================================================


DISEASE_DATABASE = {

    # ========================================================
    # APPLE
    # ========================================================

    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab",
        "cause": "Fungus Venturia inaequalis",
        "symptoms": [
            "Olive-green or brown spots on leaves",
            "Dark lesions may develop on fruit",
            "Severe infection can cause premature leaf drop"
        ],
        "treatment": [
            "Remove infected leaves",
            "Prune affected plant material",
            "Use an appropriate fungicide according to local recommendations"
        ],
        "organic_treatment": [
            "Remove fallen infected leaves",
            "Improve airflow through pruning",
            "Use approved organic fungicides where appropriate"
        ],
        "prevention": [
            "Remove fallen leaves",
            "Maintain good air circulation",
            "Avoid prolonged leaf wetness"
        ]
    },


    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot",
        "cause": "Fungus Botryosphaeria obtusa",
        "symptoms": [
            "Purple or brown leaf spots",
            "Darkening lesions",
            "Fruit may develop dark rotten areas"
        ],
        "treatment": [
            "Remove infected plant material",
            "Prune dead branches",
            "Use appropriate fungicide"
        ],
        "organic_treatment": [
            "Remove infected fruit and branches",
            "Keep the orchard clean"
        ],
        "prevention": [
            "Remove dead wood",
            "Remove mummified fruit",
            "Improve canopy ventilation"
        ]
    },


    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust",
        "cause": "Fungus Gymnosporangium species",
        "symptoms": [
            "Yellow-orange spots on leaves",
            "Orange fungal structures may appear",
            "Premature leaf loss may occur"
        ],
        "treatment": [
            "Remove severely affected material",
            "Use an appropriate fungicide"
        ],
        "organic_treatment": [
            "Improve airflow",
            "Remove infected fallen material"
        ],
        "prevention": [
            "Manage nearby alternate hosts",
            "Maintain good air circulation"
        ]
    },


    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy",
        "cause": "No disease detected",
        "symptoms": [
            "No obvious disease symptoms detected"
        ],
        "treatment": [
            "No treatment required"
        ],
        "organic_treatment": [
            "Continue normal plant care"
        ],
        "prevention": [
            "Maintain proper watering",
            "Provide adequate sunlight",
            "Monitor regularly"
        ]
    },


    # ========================================================
    # BLUEBERRY
    # ========================================================

    "Blueberry___healthy": {
        "plant": "Blueberry",
        "disease": "Healthy",
        "cause": "No disease detected",
        "symptoms": [
            "No obvious disease symptoms detected"
        ],
        "treatment": [
            "No treatment required"
        ],
        "organic_treatment": [
            "Continue normal plant care"
        ],
        "prevention": [
            "Maintain appropriate soil conditions",
            "Provide adequate sunlight",
            "Monitor regularly"
        ]
    },


    # ========================================================
    # CHERRY
    # ========================================================

    "Cherry_(including_sour)___Powdery_mildew": {
        "plant": "Cherry",
        "disease": "Powdery Mildew",
        "cause": "Powdery mildew fungi",
        "symptoms": [
            "White powder-like growth on leaves",
            "Leaf curling or distortion",
            "Young shoots may become infected"
        ],
        "treatment": [
            "Remove severely infected leaves",
            "Improve air circulation",
            "Use an appropriate fungicide"
        ],
        "organic_treatment": [
            "Remove heavily affected material",
            "Use approved organic mildew treatments"
        ],
        "prevention": [
            "Avoid overcrowding",
            "Maintain airflow",
            "Avoid excessive nitrogen"
        ]
    },


    "Cherry_(including_sour)___healthy": {
        "plant": "Cherry",
        "disease": "Healthy",
        "cause": "No disease detected",
        "symptoms": [
            "No obvious disease symptoms detected"
        ],
        "treatment": [
            "No treatment required"
        ],
        "organic_treatment": [
            "Continue normal care"
        ],
        "prevention": [
            "Maintain good airflow",
            "Monitor leaves regularly"
        ]
    },


    # ========================================================
    # CORN
    # ========================================================

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "plant": "Corn",
        "disease": "Cercospora Leaf Spot / Gray Leaf Spot",
        "cause": "Fungus Cercospora species",
        "symptoms": [
            "Long gray or brown lesions",
            "Lesions may occur along leaf veins",
            "Severe infection reduces green leaf area"
        ],
        "treatment": [
            "Use resistant varieties",
            "Apply appropriate fungicide where recommended"
        ],
        "organic_treatment": [
            "Remove crop residue where practical",
            "Improve field management"
        ],
        "prevention": [
            "Crop rotation",
            "Use resistant varieties",
            "Manage crop residue"
        ]
    },


    "Corn_(maize)___Common_rust_": {
        "plant": "Corn",
        "disease": "Common Rust",
        "cause": "Fungus Puccinia sorghi",
        "symptoms": [
            "Small reddish-brown pustules",
            "Pustules occur on leaf surfaces",
            "Severe infection can reduce photosynthesis"
        ],
        "treatment": [
            "Use resistant varieties",
            "Apply fungicide when economically justified"
        ],
        "organic_treatment": [
            "Maintain healthy crop growth",
            "Use resistant varieties"
        ],
        "prevention": [
            "Plant resistant varieties",
            "Monitor crops regularly"
        ]
    },


    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn",
        "disease": "Northern Leaf Blight",
        "cause": "Fungus Exserohilum turcicum",
        "symptoms": [
            "Long gray-green lesions",
            "Lesions become brown as disease progresses",
            "Large portions of leaves may become affected"
        ],
        "treatment": [
            "Use resistant varieties",
            "Use appropriate fungicide when recommended"
        ],
        "organic_treatment": [
            "Crop rotation",
            "Remove or manage infected crop residue"
        ],
        "prevention": [
            "Crop rotation",
            "Use resistant varieties",
            "Manage crop residue"
        ]
    },


    "Corn_(maize)___healthy": {
        "plant": "Corn",
        "disease": "Healthy",
        "cause": "No disease detected",
        "symptoms": [
            "No obvious disease symptoms detected"
        ],
        "treatment": [
            "No treatment required"
        ],
        "organic_treatment": [
            "Continue normal crop management"
        ],
        "prevention": [
            "Maintain balanced nutrition",
            "Monitor regularly"
        ]
    },


    # ========================================================
    # GRAPE
    # ========================================================

    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot",
        "cause": "Fungus Guignardia bidwellii",
        "symptoms": [
            "Brown leaf spots",
            "Dark fruit lesions",
            "Fruit may become shriveled and black"
        ],
        "treatment": [
            "Remove infected fruit",
            "Prune infected material",
            "Apply appropriate fungicide"
        ],
        "organic_treatment": [
            "Remove mummified fruit",
            "Improve canopy airflow"
        ],
        "prevention": [
            "Remove infected debris",
            "Maintain good vineyard sanitation",
            "Improve airflow"
        ]
    },


    "Grape___Esca_(Black_Measles)": {
        "plant": "Grape",
        "disease": "Esca / Black Measles",
        "cause": "Complex of wood-associated fungal pathogens",
        "symptoms": [
            "Leaf discoloration",
            "Interveinal chlorosis",
            "Dark spots may appear on fruit"
        ],
        "treatment": [
            "Remove severely affected plant material",
            "Manage infected vines according to local viticulture guidance"
        ],
        "organic_treatment": [
            "Remove affected material",
            "Maintain vineyard sanitation"
        ],
        "prevention": [
            "Use healthy planting material",
            "Avoid unnecessary trunk injuries",
            "Monitor vines regularly"
        ]
    },


    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "plant": "Grape",
        "disease": "Leaf Blight",
        "cause": "Fungal pathogen associated with Isariopsis leaf spot",
        "symptoms": [
            "Dark leaf spots",
            "Leaf browning",
            "Premature leaf drop may occur"
        ],
        "treatment": [
            "Remove severely affected leaves",
            "Improve canopy ventilation",
            "Use appropriate fungicide"
        ],
        "organic_treatment": [
            "Remove infected leaves",
            "Improve airflow"
        ],
        "prevention": [
            "Maintain vineyard sanitation",
            "Avoid prolonged leaf wetness"
        ]
    },


    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy",
        "cause": "No disease detected",
        "symptoms": [
            "No obvious disease symptoms detected"
        ],
        "treatment": [
            "No treatment required"
        ],
        "organic_treatment": [
            "Continue normal vineyard care"
        ],
        "prevention": [
            "Maintain good airflow",
            "Monitor vines regularly"
        ]
    },


    # ========================================================
    # ORANGE
    # ========================================================

    "Orange___Haunglongbing_(Citrus_greening)": {
        "plant": "Orange",
        "disease": "Huanglongbing / Citrus Greening",
        "cause": "Candidatus Liberibacter bacteria transmitted mainly by psyllids",
        "symptoms": [
            "Blotchy mottled leaves",
            "Yellowing of shoots",
            "Small or poorly colored fruit",
            "Premature fruit drop"
        ],
        "treatment": [
            "There is currently no reliable cure for infected trees",
            "Control insect vectors",
            "Remove severely infected trees according to local guidance"
        ],
        "organic_treatment": [
            "Monitor and manage psyllid populations",
            "Remove severely affected trees according to local regulations"
        ],
        "prevention": [
            "Use certified disease-free planting material",
            "Control psyllid vectors",
            "Inspect trees regularly"
        ]
    },


    # ========================================================
    # PEACH
    # ========================================================

    "Peach___Bacterial_spot": {
        "plant": "Peach",
        "disease": "Bacterial Spot",
        "cause": "Xanthomonas species",
        "symptoms": [
            "Small dark leaf spots",
            "Shot-hole appearance",
            "Fruit lesions may develop"
        ],
        "treatment": [
            "Remove severely affected material",
            "Use appropriate copper-based products where locally recommended"
        ],
        "organic_treatment": [
            "Use approved copper products according to label",
            "Remove infected debris"
        ],
        "prevention": [
            "Use healthy planting material",
            "Avoid prolonged leaf wetness",
            "Maintain airflow"
        ]
    },


    "Peach___healthy": {
        "plant": "Peach",
        "disease": "Healthy",
        "cause": "No disease detected",
        "symptoms": [
            "No obvious disease symptoms detected"
        ],
        "treatment": [
            "No treatment required"
        ],
        "organic_treatment": [
            "Continue normal care"
        ],
        "prevention": [
            "Maintain good airflow",
            "Monitor leaves regularly"
        ]
    },


    # ========================================================
    # PEPPER
    # ========================================================

    "Pepper,_bell___Bacterial_spot": {
        "plant": "Bell Pepper",
        "disease": "Bacterial Spot",
        "cause": "Xanthomonas species",
        "symptoms": [
            "Small dark leaf spots",
            "Yellow halos around lesions",
            "Fruit may develop raised spots"
        ],
        "treatment": [
            "Remove severely infected leaves",
            "Use appropriate copper-based bactericide",
            "Avoid overhead irrigation"
        ],
        "organic_treatment": [
            "Use approved copper products",
            "Remove infected plant material"
        ],
        "prevention": [
            "Use disease-free seeds",
            "Rotate crops",
            "Keep foliage dry"
        ]
    },


    "Pepper,_bell___healthy": {
        "plant": "Bell Pepper",
        "disease": "Healthy",
        "cause": "No disease detected",
        "symptoms": [
            "No obvious disease symptoms detected"
        ],
        "treatment": [
            "No treatment required"
        ],
        "organic_treatment": [
            "Continue normal plant care"
        ],
        "prevention": [
            "Maintain good airflow",
            "Avoid excessive leaf wetness"
        ]
    },


    # ========================================================
    # POTATO
    # ========================================================

    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight",
        "cause": "Fungus Alternaria solani",
        "symptoms": [
            "Brown circular leaf lesions",
            "Concentric ring patterns",
            "Older leaves are often affected first"
        ],
        "treatment": [
            "Remove severely infected leaves",
            "Apply appropriate fungicide",
            "Avoid prolonged leaf wetness"
        ],
        "organic_treatment": [
            "Remove infected foliage",
            "Improve airflow",
            "Maintain plant nutrition"
        ],
        "prevention": [
            "Crop rotation",
            "Use healthy seed potatoes",
            "Avoid overhead irrigation"
        ]
    },


    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight",
        "cause": "Oomycete Phytophthora infestans",
        "symptoms": [
            "Water-soaked leaf lesions",
            "Brown or black rapidly expanding areas",
            "White growth may appear under humid conditions"
        ],
        "treatment": [
            "Remove severely infected plant material",
            "Apply appropriate fungicide promptly",
            "Prevent prolonged leaf wetness"
        ],
        "organic_treatment": [
            "Remove infected plant material",
            "Improve airflow",
            "Use approved organic products where appropriate"
        ],
        "prevention": [
            "Use healthy seed potatoes",
            "Monitor weather conditions",
            "Avoid prolonged leaf wetness"
        ]
    },


    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy",
        "cause": "No disease detected",
        "symptoms": [
            "No obvious disease symptoms detected"
        ],
        "treatment": [
            "No treatment required"
        ],
        "organic_treatment": [
            "Continue normal crop care"
        ],
        "prevention": [
            "Use healthy seed",
            "Practice crop rotation",
            "Monitor regularly"
        ]
    },


    # ========================================================
    # RASPBERRY
    # ========================================================

    "Raspberry___healthy": {
        "plant": "Raspberry",
        "disease": "Healthy",
        "cause": "No disease detected",
        "symptoms": [
            "No obvious disease symptoms detected"
        ],
        "treatment": [
            "No treatment required"
        ],
        "organic_treatment": [
            "Continue normal plant care"
        ],
        "prevention": [
            "Maintain good airflow",
            "Remove damaged plant material"
        ]
    },


    # ========================================================
    # SOYBEAN
    # ========================================================

    "Soybean___healthy": {
        "plant": "Soybean",
        "disease": "Healthy",
        "cause": "No disease detected",
        "symptoms": [
            "No obvious disease symptoms detected"
        ],
        "treatment": [
            "No treatment required"
        ],
        "organic_treatment": [
            "Continue normal crop care"
        ],
        "prevention": [
            "Maintain balanced nutrition",
            "Monitor crop regularly"
        ]
    },


    # ========================================================
    # SQUASH
    # ========================================================

    "Squash___Powdery_mildew": {
        "plant": "Squash",
        "disease": "Powdery Mildew",
        "cause": "Powdery mildew fungi",
        "symptoms": [
            "White powder-like patches",
            "Leaf yellowing",
            "Reduced photosynthetic area"
        ],
        "treatment": [
            "Remove severely infected leaves",
            "Improve airflow",
            "Use an appropriate fungicide"
        ],
        "organic_treatment": [
            "Use approved organic mildew treatments",
            "Remove heavily infected leaves"
        ],
        "prevention": [
            "Avoid overcrowding",
            "Maintain airflow",
            "Avoid excessive humidity"
        ]
    },


    # ========================================================
    # STRAWBERRY
    # ========================================================

    "Strawberry___Leaf_scorch": {
        "plant": "Strawberry",
        "disease": "Leaf Scorch",
        "cause": "Fungal pathogen associated with leaf scorch",
        "symptoms": [
            "Dark purple or brown leaf spots",
            "Spots may enlarge",
            "Leaves can appear scorched"
        ],
        "treatment": [
            "Remove severely affected leaves",
            "Improve plant spacing",
            "Use appropriate fungicide if recommended"
        ],
        "organic_treatment": [
            "Remove infected leaves",
            "Improve airflow"
        ],
        "prevention": [
            "Avoid prolonged leaf wetness",
            "Maintain sanitation",
            "Use healthy planting material"
        ]
    },


    "Strawberry___healthy": {
        "plant": "Strawberry",
        "disease": "Healthy",
        "cause": "No disease detected",
        "symptoms": [
            "No obvious disease symptoms detected"
        ],
        "treatment": [
            "No treatment required"
        ],
        "organic_treatment": [
            "Continue normal plant care"
        ],
        "prevention": [
            "Maintain good airflow",
            "Avoid excessive moisture"
        ]
    },


    # ========================================================
    # TOMATO
    # ========================================================

    "Tomato___Bacterial_spot": {
        "plant": "Tomato",
        "disease": "Bacterial Spot",
        "cause": "Xanthomonas species",
        "symptoms": [
            "Small dark leaf spots",
            "Yellow halos",
            "Fruit may develop dark lesions"
        ],
        "treatment": [
            "Remove severely infected leaves",
            "Use appropriate copper-based bactericide",
            "Avoid overhead watering"
        ],
        "organic_treatment": [
            "Use approved copper products",
            "Remove infected material"
        ],
        "prevention": [
            "Use disease-free seed",
            "Rotate crops",
            "Keep leaves dry"
        ]
    },


    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight",
        "cause": "Fungus Alternaria solani",
        "symptoms": [
            "Brown circular leaf spots",
            "Concentric rings",
            "Lower leaves are often affected first"
        ],
        "treatment": [
            "Remove infected leaves",
            "Apply appropriate fungicide",
            "Avoid excessive leaf wetness"
        ],
        "organic_treatment": [
            "Remove affected leaves",
            "Improve airflow",
            "Use approved organic fungicides"
        ],
        "prevention": [
            "Crop rotation",
            "Mulch soil to reduce splash",
            "Avoid overhead irrigation"
        ]
    },


    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight",
        "cause": "Oomycete Phytophthora infestans",
        "symptoms": [
            "Water-soaked lesions",
            "Rapidly expanding brown or black areas",
            "White growth may appear under humid conditions"
        ],
        "treatment": [
            "Remove severely infected leaves",
            "Use appropriate fungicide promptly",
            "Improve airflow"
        ],
        "organic_treatment": [
            "Remove infected material",
            "Reduce leaf wetness",
            "Use approved organic products where appropriate"
        ],
        "prevention": [
            "Avoid prolonged leaf wetness",
            "Provide good air circulation",
            "Monitor plants frequently"
        ]
    },


    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Tomato Leaf Mold",
        "cause": "Fungus Passalora fulva",
        "symptoms": [
            "Yellow patches on upper leaf surfaces",
            "Olive or gray fungal growth underneath leaves",
            "Leaves may dry and fall"
        ],
        "treatment": [
            "Remove infected leaves",
            "Improve ventilation",
            "Use appropriate fungicide"
        ],
        "organic_treatment": [
            "Remove affected leaves",
            "Improve greenhouse ventilation",
            "Reduce humidity"
        ],
        "prevention": [
            "Improve airflow",
            "Avoid excessive humidity",
            "Keep foliage dry"
        ]
    },


    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot",
        "cause": "Fungus Septoria lycopersici",
        "symptoms": [
            "Small circular leaf spots",
            "Dark borders",
            "Tiny black fruiting structures may occur"
        ],
        "treatment": [
            "Remove infected lower leaves",
            "Apply appropriate fungicide",
            "Avoid overhead watering"
        ],
        "organic_treatment": [
            "Remove infected foliage",
            "Use mulch to reduce soil splash"
        ],
        "prevention": [
            "Crop rotation",
            "Avoid overhead watering",
            "Remove infected plant debris"
        ]
    },


    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "plant": "Tomato",
        "disease": "Two-Spotted Spider Mite",
        "cause": "Two-spotted spider mite Tetranychus urticae",
        "symptoms": [
            "Fine yellow or pale speckling",
            "Leaf bronzing",
            "Fine webbing may appear",
            "Severe infestation can cause leaf drop"
        ],
        "treatment": [
            "Wash plants with water where appropriate",
            "Use an appropriate miticide",
            "Remove heavily affected leaves"
        ],
        "organic_treatment": [
            "Use insecticidal soap where appropriate",
            "Encourage natural predators",
            "Use horticultural oils according to label"
        ],
        "prevention": [
            "Monitor leaf undersides",
            "Avoid severe plant stress",
            "Maintain appropriate humidity"
        ]
    },


    "Tomato___Target_Spot": {
        "plant": "Tomato",
        "disease": "Target Spot",
        "cause": "Fungus Corynespora cassiicola",
        "symptoms": [
            "Circular brown leaf lesions",
            "Concentric target-like rings",
            "Leaves may yellow and drop"
        ],
        "treatment": [
            "Remove infected leaves",
            "Improve airflow",
            "Use appropriate fungicide"
        ],
        "organic_treatment": [
            "Remove infected foliage",
            "Reduce leaf wetness",
            "Improve ventilation"
        ],
        "prevention": [
            "Avoid overhead watering",
            "Maintain airflow",
            "Remove infected debris"
        ]
    },


    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "cause": "Tomato yellow leaf curl virus, commonly transmitted by whiteflies",
        "symptoms": [
            "Upward curling leaves",
            "Yellowing of leaf margins",
            "Stunted plant growth",
            "Reduced fruit production"
        ],
        "treatment": [
            "Remove severely infected plants",
            "Control whitefly populations",
            "Use virus-free planting material"
        ],
        "organic_treatment": [
            "Use approved methods to control whiteflies",
            "Remove severely infected plants"
        ],
        "prevention": [
            "Control whiteflies",
            "Use healthy seedlings",
            "Remove infected plants promptly"
        ]
    },


    "Tomato___Tomato_mosaic_virus": {
        "plant": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "cause": "Tomato mosaic virus",
        "symptoms": [
            "Mosaic light and dark green patterns",
            "Leaf distortion",
            "Reduced plant growth"
        ],
        "treatment": [
            "There is no curative treatment for infected plants",
            "Remove infected plants",
            "Disinfect tools"
        ],
        "organic_treatment": [
            "Remove infected plants",
            "Maintain strict sanitation"
        ],
        "prevention": [
            "Use healthy seed",
            "Disinfect tools",
            "Avoid handling plants after touching infected material"
        ]
    },


    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy",
        "cause": "No disease detected",
        "symptoms": [
            "No obvious disease symptoms detected"
        ],
        "treatment": [
            "No treatment required"
        ],
        "organic_treatment": [
            "Continue normal plant care"
        ],
        "prevention": [
            "Maintain proper watering",
            "Provide adequate sunlight",
            "Maintain good airflow",
            "Monitor regularly"
        ]
    }

}


# ============================================================
# LOOKUP FUNCTION
# ============================================================

def get_disease(disease_name):

    if not disease_name:

        print(
            "WARNING: Empty disease name"
        )

        return None


    original_name = str(
        disease_name
    ).strip()


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

        name = str(
            name
        ).strip()

        # Remove duplicate spaces
        name = " ".join(
            name.split()
        )

        # Normalize Corn naming
        name = name.replace(
            "Corn_(maize)",
            "Corn"
        )

        name = name.replace(
            "Corn_(Maize)",
            "Corn"
        )

        # Remove trailing underscore
        name = name.rstrip("_")

        return name.lower()


    normalized_prediction = normalize(
        original_name
    )


    print(
        "Normalized prediction:",
        repr(
            normalized_prediction
        )
    )


    # ========================================================
    # EXACT / NORMALIZED MATCH
    # ========================================================

    for key, value in DISEASE_DATABASE.items():

        if (
            normalize(key)
            ==
            normalized_prediction
        ):

            print(
                "MATCH:",
                key
            )

            print("=" * 60)

            return value


    # ========================================================
    # PLANT + DISEASE MATCH
    # ========================================================

    if "___" in original_name:

        prediction_plant, prediction_disease = (
            original_name.split(
                "___",
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


            if (
                prediction_plant
                ==
                database_plant
                and
                prediction_disease
                ==
                database_disease
            ):

                print(
                    "MATCH:",
                    key
                )

                print("=" * 60)

                return value


    # ========================================================
    # SPECIAL CORN MATCHING
    # ========================================================

    prediction_clean = (
        normalized_prediction
        .replace(
            "corn_(maize)",
            "corn"
        )
    )


    for key, value in DISEASE_DATABASE.items():

        key_clean = normalize(
            key
        )

        key_clean = (
            key_clean
            .replace(
                "corn_(maize)",
                "corn"
            )
        )


        if (
            key_clean
            ==
            prediction_clean
        ):

            print(
                "MATCH AFTER CORN NORMALIZATION:",
                key
            )

            print("=" * 60)

            return value


    # ========================================================
    # PARTIAL MATCH
    # ========================================================

    for key, value in DISEASE_DATABASE.items():

        key_normalized = normalize(
            key
        )


        if (
            normalized_prediction
            in key_normalized
            or
            key_normalized
            in normalized_prediction
        ):

            print(
                "PARTIAL MATCH:",
                key
            )

            print("=" * 60)

            return value


    # ========================================================
    # NO MATCH
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