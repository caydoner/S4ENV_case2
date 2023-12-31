import cv2
import numpy as np
import sqlite3
import pandas as pd


def add_image_to_database(image_path, db_path, table_name):
    # Read the image from file
    image = cv2.imread(image_path)

    # Convert the image to bytes
    success, encoded_image = cv2.imencode(".jpg", image)
    if success:
        # Get the byte array from the encoded image data
        image_bytes = np.array(encoded_image).tobytes()

        # Now you can use 'image_bytes' to store the image data or send it over networks, etc.
    else:
        print("Failed to encode the image.")
    # Connect to the SQLite database
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Create the table if it doesn't exist
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            image_id INTEGER PRIMARY KEY,
            image BLOB
        )
    """)

    # Insert the image into the table
    cursor.execute(f"INSERT INTO {table_name} (image) VALUES (?)", (sqlite3.Binary(image_bytes),))

    # Commit the changes and close the connection
    connection.commit()
    connection.close()


def read_image_from_database(db_path, table_name, image_id):
    # Connect to the SQLite database
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    # Retrieve the image data from the table
    cursor.execute(f"SELECT image FROM {table_name} WHERE image_id = ?", (image_id,))
    image_blob = cursor.fetchone()

    # Close the connection
    connection.close()

    if image_blob:
        # Convert the image data back to bytes and decode using numpy
        image_data = np.frombuffer(image_blob[0], dtype=np.uint8)
        
        # Decode the image using OpenCV
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is not None:
            cv2.imshow("Image from Database", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return image
    else:
        print("Image not found.")
        return None


def calculate_roughness(cv2img):
    gray_image = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    roughness_index = np.mean(gradient_magnitude)
    return roughness_index


def calculate_brightness(cv2img):
    gray_image = cv2.cvtColor(cv2img, cv2.COLOR_BGR2GRAY)
    brightness_index = np.mean(gray_image)
    return brightness_index


def capture_image_from_webcam(output_path):
    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Couldn't open the webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't capture a frame.")
            break

        cv2.imshow("Webcam Stream", frame)

        # Press 'c' to capture the current frame
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(output_path, frame)
            print(f"Image captured and saved as {output_path}")
            break
        elif key == 27:  # Press 'Esc' key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

def sql_2_df(db,table_name):

    # SQLite veritabanına bağlan
    con = sqlite3.connect(db)

    # Pandas ile SQL sorgusunu kullanarak veriyi oku
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, con)

    # Bağlantıyı kapat
    con.close()
    return df