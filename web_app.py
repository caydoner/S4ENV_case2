from utilities import *
import streamlit as st
import os


def main():
    st.title("Webcam Capture and Analysis")
    os.makedirs("images",exist_ok=True)
    os.makedirs("data",exist_ok=True)
    image_path = r"images/image.jpg"  # Provide the path to your image
    db_path = r"data/database.db"  # Provide the path to your SQLite database
    table_name = r"camur_bilgi"  # Provide the name of the table in the database
    picture=st.camera_input("AAT Çamur Fotoğrafı")
    if picture is not None:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(image_path,cv2_img)
        roughness_index = calculate_roughness(cv2_img)
        st.write(f"Roughness Index: {roughness_index:.2f}")
        brightness_index = calculate_brightness(cv2_img)
        st.write(f"Brightness Index: {brightness_index:.2f}")
        with st.form("AAT Çamur Bilgisi",clear_on_submit=True):
            aat=st.text_input(label='AAT Adı',disabled=False,key='aat',value="")
            numune=st.text_input(label='Numune Adı',disabled=False,key='numune',value="")
            tarih=st.date_input(label='Numune Tarihi',disabled=False,key='tarih',value=None)
            nem=st.slider(label="Lütfen Nem Değerini Giriniz",min_value=0,max_value=1500,value=0,disabled=False,key='nem')
            kaydet=st.form_submit_button('Kaydet',disabled=False,)
            if all([picture,aat,numune,tarih,nem]):
                if kaydet:
                    add_image_to_database(image_path, db_path, table_name)
                    st.success("Görüntü bilgileri Kaydedildi")
                    picture=None
   
            else:
                st.warning("Lütfen Eksik Bilgileri Tamamlayınız!" )
    else:
        st.empty()





if __name__ == "__main__":
    main()