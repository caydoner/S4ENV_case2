import cv2
import numpy as np
import sqlite3
import pandas as pd
import streamlit as st
import sys
import rembg
from io import BytesIO
from st_aggrid import AgGrid,GridOptionsBuilder,GridUpdateMode,JsCode,AgGridReturn,DataReturnMode,AgGridTheme

st.set_page_config(layout="centered")

if 'secim' not in st.session_state:
    st.session_state['secim'] = None





col1,col3,col5=st.columns(3)
with col1:
     st.image(".streamlit/tubitakmam.jpg",width=50,use_column_width=False)
with col3:
     st.image(".streamlit/Smart4EnvLogo.png",width=150,use_column_width=False)
with col5:
     st.image(".streamlit/smart4envbacky.png",width=150,use_column_width=True)

secenek=['Yerel Bilgisayardan Aktar','Kameradan Aktar']
secim=st.radio('ÇAMUR NUMUNE FOTOĞRAFI AKTARMA ',secenek)


db_file="database.db"
table_name='olcumler'


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


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


def create_table(db_file,table_name):
    """ create a table from the create_table_sql statement
    :param the_conn: Connection object
    """
    the_conn=sqlite3.connect(database=db_file)
    c = the_conn.cursor()
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name}(
            image_id INTEGER PRIMARY KEY,
            tarih TEXT,
            numune_adi TEXT,
            nem INTEGER, 
            puruzluluk FLOAT,
            parlaklik FLOAT,
            image BLOB)
            """)
    the_conn.commit()
    the_conn.close()


def add_data(db_file,the_table, kayit):
    the_conn = sqlite3.connect(database=db_file)
    cur = the_conn.cursor()
    cur.execute(f"""INSERT INTO {the_table}(tarih,numune_adi,nem,puruzluluk,parlaklik,image) VALUES (?,?,?,?,?,?) """, kayit)
    the_conn.commit()
    the_conn.close()

def kayit_sil(db_file,the_table, kayit):
    the_conn = sqlite3.connect(database=db_file)
    cur = the_conn.cursor()
    cur.execute(f"""DELETE FROM {the_table} WHERE image_id=?""", kayit)
    the_conn.commit()
    the_conn.close()

def delete_table(db_file,the_table):
    the_conn = sqlite3.connect(database=db_file)
    cur = the_conn.cursor()
    cur.execute(f"""DROP TABLE IF EXISTS {the_table}""")
    the_conn.commit()
    the_conn.close()


def read_image_from_database(db_file, table_name,numune_adi):
    # Connect to the SQLite database
    the_conn = sqlite3.connect(database=db_file)
    cursor = the_conn.cursor()
    # Retrieve the image data from the table
    cursor.execute(f"SELECT image FROM {table_name} WHERE numune_adi = ?", (numune_adi,))
    image_blob = cursor.fetchone()

    # Close the connection
    the_conn.close()

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



def sql_2_df(db_file,table_name):
    # SQLite veritabanına bağlan
    con = create_dbconnection(db_file)
    # Pandas ile SQL sorgusunu kullanarak veriyi oku
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, con)
    # Bağlantıyı kapat
    con.close()
    return df


def add_column():
     ncols=st.session_state['ncount']
     columns=st.columns(ncols)
     for i in range(ncols):
          columns[i]=st.empty()


def calc_and_save_picture_data(picture):
    if picture is not None:
        bytes_data = picture.getvalue()
        bytes_data_rbg=rembg.remove(bytes_data)
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data_rbg, np.uint8), cv2.IMREAD_COLOR)
        roughness_index = calculate_roughness(cv2_img)
        brightness_index = calculate_brightness(cv2_img)
        with st.form("Numune Fotoğraflari",clear_on_submit=True):
            #st.write(f"Roughness Index: {roughness_index:.2f}")
            #st.write(f"Brightness Index: {brightness_index:.2f}")
            numune_adi=st.text_input(label='Numune Adı',placeholder="Numune Adını Giriniz...")
            tarih=st.date_input(label='Tarih',disabled=False,key='tarih',value=None,format="DD/MM/YYYY")
            nem=st.number_input('Nem Değeri',min_value=0.0,max_value=100.0,placeholder="Nem Değerini Giriniz...")
            #st.write(all([picture,roughness_index,brightness_index,numune_adi]))
            kaydet=st.form_submit_button('Kaydet',disabled=False,)
            if all([picture,roughness_index,brightness_index,numune_adi]):
                if kaydet:
                    add_data(db_file=db_file,the_table=table_name,kayit=(tarih,numune_adi,nem,roughness_index,brightness_index,sqlite3.Binary(bytes_data_rbg)))
                    st.success("Numune Kaydedildi")

                    del picture
                    del roughness_index
                    del brightness_index
                    del numune_adi
                    del bytes_data_rbg
            else:
                st.warning("Lütfen Numune Adını Giriniz!" )



def main():
    # if st.button(label='Verileri Temizle',disabled=False):
    #     delete_table(db_file=db_file,the_table=table_name)
    #     st.success("Veritabanı temizlendi.")
    create_table(db_file=db_file,table_name=table_name)
    local_css('style.css')
    #st.dataframe(sql_2_df(db_file=db_file,the_table=table_name))
    #ncount=st.selectbox(label="NUMUNE SAYISI",options=[1,2,3,4,5])


    if secim=="Yerel Bilgisayardan Aktar":
        picture = st.file_uploader("NUMUNE FOTOĞRAFI", type=["jpg", "jpeg", "png"],key="file")
    elif secim=="Kameradan Aktar":
        picture=st.camera_input("NUMUNE FOTOĞRAFI",key="camfile")
    else:
        picture=None

    calc_and_save_picture_data(picture)
    df = sql_2_df(db_file, table_name)

    builder = GridOptionsBuilder.from_dataframe(df.iloc[:, :-1])
    builder.configure_selection(selection_mode='single',use_checkbox=True)
    builder.configure_pagination()
    go = builder.build()
    ag = AgGrid(
        df.iloc[:, :-1],
        gridOptions=go,
        fit_columns_on_grid_load=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        key=None,
        theme=AgGridTheme.BALHAM)
    if ag.selected_rows:
        if st.button('Seçili Kaydı Sil'):
            with st.spinner("Kayıt Siliniyor..."):
                kayit_sil(db_file, table_name, [ag['selected_rows'][0]['image_id']])
                st.success("Kayıt Silindi.")
                st.rerun()






    #st.markdown(f"""AgGrid({df.iloc[:, :-1]}, gridOptions={go},fit_columns_on_grid_load=True, reload_data={reload_data}, height={height})""")
    #grid_table=AgGrid(df.iloc[:, :-1], gridOptions=go,fit_columns_on_grid_load=True, key=None,update_mode = GridUpdateMode.MODEL_CHANGED, reload_data=True)
    # selections=grid_table['selected_rows']
    # st.write(selections)
    # sil=st.button("Seçili Kaydı Sil")
    # if len(selections)==1 and sil:
    #     kayit_sil(db_file,table_name,[selections[0]['image_id']])
    #     st.success("Kayıt Silindi.")
    #     grid_table.data=df.iloc[:, :-1]
    # else:
    #     st.warning("Lütfen Silmek İstediğiniz Kaydı Seçiniz...")


    # #EXCEL FORMATINA DONUŞTUR VE INDIR
    # output = BytesIO()
    # with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
    #     # Write each dataframe to a different worksheet.
    #     df.iloc[:,0:-1].to_excel(writer, sheet_name='Sheet1', index=False)
    #
    # st.download_button(
    #     label="Excel olarak indir",
    #     data=output.getvalue(),
    #     file_name='camur_numune.xlsx',
    #     mime='application/vnd.ms-excel'
    # )

    #NEM DURUMUNA GÖRE SIRALA
    ncount=df.shape[0]
    df["opt"]=df.image_id.astype("string") +"-->"+ df.numune_adi
    fotos=st.multiselect(label="NUMUNE SEÇİNİZ",options=df.opt.unique().tolist())
    numune_pruz={}
    if ncount>0 and len(fotos)>0:
        cols=st.columns(len(fotos))
        for idx,ad in enumerate(fotos):
            numune_adi=ad.split("-->")[1]
            puruzluluk=df.loc[df.numune_adi==numune_adi].puruzluluk.values[0]
            cols[idx].write(numune_adi)
            cols[idx].image(read_image_from_database(db_file=db_file,table_name=table_name,numune_adi=numune_adi))
            #cols[i].image(read_image_from_database(db_file=db_file,table_name=table_name,imageid=i+1))
            #cols[idx].checkbox("SEÇ",value=False,key=)
            #cols[idx].write(f"PÜRÜZLÜLÜK:{round(df.loc[df.numune_adi==numune_adi].puruzluluk.values[0],2)}")
            numune_pruz[numune_adi]=puruzluluk

    if st.button("Seçilen Numuneleri Nem İçeriğine Göre Sırala"):
        liste=[v[0] for v in sorted(numune_pruz.items(),key=lambda x:x[1])]
        colms=st.columns(len(liste))
        for i in range(len(liste)):
            colms[i].write(liste[i])
            colms[i].image(read_image_from_database(db_file=db_file,table_name=table_name,numune_adi=liste[i]))



               
if __name__ == "__main__":
    main()

