import sqlite3
import os


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


DATABASE = os.path.join(
    BASE_DIR,
    "plantai.db"
)



def get_connection():

    conn = sqlite3.connect(
        DATABASE
    )

    conn.row_factory = sqlite3.Row

    return conn





def create_tables():

    conn = get_connection()

    cursor = conn.cursor()


    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions
        (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            image TEXT,

            plant TEXT,

            disease TEXT,

            confidence REAL,

            severity TEXT,

            affected_area REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )


    conn.commit()

    conn.close()



if __name__ == "__main__":

    create_tables()

    print("Database created successfully")

    print(DATABASE)