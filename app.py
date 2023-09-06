import cv2
import numpy as np
import sqlite3
import pandas as pd
import streamlit as st
import sys

db_file="database.db"
table_name='olcumler'

def create_dbconnection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    the_conn = sqlite3.connect(db_file, check_same_thread=False)
    if the_conn is not None:
        return the_conn
    else:
        print("Bağlantı Kurulamıyor...")
        sys.exit


def create_table(the_conn,table_name):
    """ create a table from the create_table_sql statement
    :param the_conn: Connection object
    """
    c = the_conn.cursor()
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name}(
            image_id INTEGER PRIMARY KEY, 
            puruzluluk float,
            parlaklik float,
            tarih text
            nem float,
            image BLOB)
            """)

def add_data(the_conn,the_table, kayit):
    cur = the_conn.cursor()
    cur.execute(f"""INSERT INTO {the_table}(puruzluluk,parlaklik,tarih,image) VALUES(?,?,?,?) """, kayit)
    the_conn.commit()
    the_conn.close()


def read_image_from_database(db_file, table_name,imageid):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    # Retrieve the image data from the table
    cursor.execute(f"SELECT image FROM {table_name} WHERE image_id = ?", (imageid,))
    image_blob = cursor.fetchone()

    # Close the connection
    conn.close()

    if image_blob:
        # Convert the image data back to bytes and decode using numpy
        image_data = np.frombuffer(image_blob[0], dtype=np.uint8)
        
        # Decode the image using OpenCV
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        # if image is not None:
        #     cv2.imshow("Image from Database", image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
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


def main():
    st.title("ÇAMUR ANALİZİ")
    # image_path = r"image.jpg"  # Provide the path to your image
    # db_path = r"database.db"  # Provide the path to your SQLite database
    # table_name = r"camur_bilgi"  # Provide the name of the table in the database
    picture=st.camera_input("AAT Çamur Fotoğrafı")
    if picture is not None:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        roughness_index = calculate_roughness(cv2_img)
        brightness_index = calculate_brightness(cv2_img)
        with st.form("AAT Çamur Bilgisi",clear_on_submit=True):
            st.write(f"Roughness Index: {roughness_index:.2f}")
            st.write(f"Brightness Index: {brightness_index:.2f}")
            #aat=st.text_input(label='AAT Adı',disabled=False,key='aat',value="")
            #numune=st.text_input(label='Numune Adı',disabled=False,key='numune',value="")
            tarih=st.date_input(label='Tarih',disabled=False,key='tarih',value=None,format="DD/MM/YYYY")
            nem=st.slider(label="Nem Değeri",min_value=0,max_value=1500,value=0,disabled=False,key='nem')
            kaydet=st.form_submit_button('Kaydet',disabled=False,)
            if all([picture,roughness_index,brightness_index,tarih]):
                if kaydet:
                    conn = create_dbconnection(db_file=db_file)
                    create_table(conn,table_name=table_name)
                    add_data(the_conn=conn,the_table=table_name,kayit=(roughness_index,brightness_index,tarih,sqlite3.Binary(bytes_data)))
                    st.success("Görüntü bilgileri Kaydedildi")
                    picture=None
   
            else:
                st.warning("Lütfen Eksik Bilgileri Tamamlayınız!" )
    else:
        st.empty()
    conn = create_dbconnection(db_file=db_file)
    create_table(the_conn=conn,table_name=table_name)
    df=sql_2_df(db=db_file,table_name=table_name)
    col1,col2,col3,col4=st.columns(4)
    with col1:
        if df.shape[0]%4==1 or df.image_id.values[0]==1:
            for i in range(1,df.shape[0]+1,4):
                st.write(f"IMAGE_ID:{i}")
                st.image(read_image_from_database(db_file=db_file,table_name=table_name,imageid=i))
                st.write(f"PÜRÜZLÜLÜK:{round(df.loc[df.image_id==i].puruzluluk.values[0],2)}")
    with col2:
        if df.shape[0]%4==2 or df.image_id.values[1]==2:
            for i in range(2,df.shape[0]+1,4):
                st.write(f"IMAGE_ID:{i}")
                st.image(read_image_from_database(db_file=db_file,table_name=table_name,imageid=i))
                st.write(f"PÜRÜZLÜLÜK:{round(df.loc[df.image_id==i].puruzluluk.values[0],2)}")      
    with col3:
        if df.shape[0]%4==3 or df.image_id.values[2]==3:
            for i in range(3,df.shape[0]+1,4):
                st.write(f"IMAGE_ID:{i}")
                st.image(read_image_from_database(db_file=db_file,table_name=table_name,imageid=i))
                st.write(f"PÜRÜZLÜLÜK:{round(df.loc[df.image_id==i].puruzluluk.values[0],2)}")
    with col4:
        if df.shape[0]%4==0 or df.image_id.values[3]==4:
            for i in range(4,df.shape[0]+1,4):
                st.write(f"IMAGE_ID:{i}")
                st.image(read_image_from_database(db_file=db_file,table_name=table_name,imageid=i))
                st.write(f"PÜRÜZLÜLÜK:{round(df.loc[df.image_id==i].puruzluluk.values[0],2)}")     
    conn.close()
    st.dataframe(df)



if __name__ == "__main__":
    main()
