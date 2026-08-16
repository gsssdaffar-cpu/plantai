from database import get_connection



def save_prediction(
        image,
        plant,
        disease,
        confidence,
        severity,
        affected_area
):

    conn = get_connection()

    cursor = conn.cursor()


    cursor.execute(

        """
        INSERT INTO predictions
        (
            image,
            plant,
            disease,
            confidence,
            severity,
            affected_area
        )

        VALUES (?,?,?,?,?,?)

        """,

        (
            image,
            plant,
            disease,
            confidence,
            severity,
            affected_area
        )

    )


    conn.commit()

    conn.close()






def get_history(search=""):


    conn = get_connection()

    cursor = conn.cursor()



    if search:


        cursor.execute(

            """

            SELECT *

            FROM predictions

            WHERE disease LIKE ?

            ORDER BY id DESC

            """,

            (
                "%" + search + "%",
            )

        )


    else:


        cursor.execute(

            """

            SELECT *

            FROM predictions

            ORDER BY id DESC

            """

        )



    rows = cursor.fetchall()

    conn.close()


    return rows







def dashboard_stats():


    conn = get_connection()

    cursor = conn.cursor()



    cursor.execute(

        "SELECT COUNT(*) FROM predictions"

    )


    total = cursor.fetchone()[0]




    cursor.execute(

        """

        SELECT COUNT(*)

        FROM predictions

        WHERE severity='Critical'

        """

    )


    critical = cursor.fetchone()[0]





    cursor.execute(

        """

        SELECT COUNT(*)

        FROM predictions

        WHERE severity='Moderate'

        """

    )


    moderate = cursor.fetchone()[0]






    cursor.execute(

        """

        SELECT COUNT(*)

        FROM predictions

        WHERE severity='Mild'

        """

    )


    mild = cursor.fetchone()[0]




    conn.close()



    return {


        "total": total,

        "critical": critical,

        "moderate": moderate,

        "mild": mild


    }








def recent_predictions():


    conn = get_connection()

    cursor = conn.cursor()



    cursor.execute(

        """

        SELECT *

        FROM predictions

        ORDER BY id DESC

        LIMIT 5

        """

    )



    rows = cursor.fetchall()


    conn.close()



    return rows