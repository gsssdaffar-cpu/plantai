# severity_advice.py


def get_severity_advice(severity):


    advice = {


        "Mild": {

            "message":
            "Disease is at an early stage. Monitor the plant regularly.",


            "actions": [

                "Remove slightly infected leaves",

                "Use organic spray like neem oil",

                "Maintain proper sunlight and airflow"

            ]

        },




        "Moderate": {


            "message":
            "Disease has progressed. Immediate treatment is recommended.",


            "actions": [

                "Remove infected plant parts",

                "Apply recommended fungicide",

                "Improve irrigation and ventilation"

            ]

        },





        "Critical": {


            "message":
            "Severe infection detected. Immediate action required.",


            "actions": [

                "Remove severely infected leaves",

                "Apply suitable disease control treatment",

                "Separate infected plants to prevent spread"

            ]

        }



    }



    return advice.get(

        severity,

        {

            "message":
            "Unable to determine severity advice.",

            "actions":[]

        }

    )