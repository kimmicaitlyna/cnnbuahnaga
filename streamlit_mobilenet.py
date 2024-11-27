import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

#load model
#sesuaikan dengan path model yang telah di dump dalam format .h5
model = r'model_mobilenet.h5'
class_names = ['Matang', 'Mentah'] #nama kelas yang akan diprediksi

#fungsi untuk memproses dan mengklasifikasikan gambar
def classify_image(image_path):
    try:
        #memproses gambar
        input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180)) #mengubah ukuran model
        input_image_array = tf.keras.utils.img_to_array(input_image) #mengubah gambar menjadi array
        input_image_exp_dim = tf.expand_dims(input_image_array, 0) #menambahkan dimensi batch

        #melakukan prediksi menggunakan model
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])  #menggunakan softmax untuk mendapatkan probabilitas
        
        #mendapatkan indeks kelas dengan probabilitas tertinggi
        class_idx = np.argmax(result)
        confidence_scores = result.numpy() #mengubah ke format numpy array
        return class_names[class_idx], confidence_scores 
    except Exception as e:
        return "Error", str(e)

#fungsi untuk membuat progress bar menggunakan HTML
def custom_progress_bar(confidence, color1, color2):
    percentage1 = confidence[0] * 100  #confidence untuk kelas 0 yaitu Matang
    percentage2 = confidence[1] * 100  #confidence untuk kelas 1 yaitu Mentah
    
    #HTML untuk progress bar
    progress_html = f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; overflow: hidden; width: 100%; font-size: 14px;">
        <div style="width: {percentage1:.2f}%; background: {color1}; color: white; text-align: center; height: 24px; float: left;">
            {percentage1:.2f}%
        </div>
        <div style="width: {percentage2:.2f}%; background: {color2}; color: white; text-align: center; height: 24px; float: left;">
            {percentage2:.2f}%
        </div>
    </div>
    """
    st.sidebar.markdown(progress_html, unsafe_allow_html=True)


st.title("Prediksi Kematangan Buah Naga - XXXX")  # 4 digit npm terakhir

#untuk mengunggah bebrapa gambar
uploaded_files = st.file_uploader("Unggah Gambar (Beberapa diperbolehkan)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

#sidebar untuk tombol prediksi dan progress bar
if st.sidebar.button("Prediksi"):
    if uploaded_files:
        st.sidebar.write("### Hasil Prediksi")
        for uploaded_file in uploaded_files:
            #simpan file gambar sementara
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            #memproses prediksi gambar
            label, confidence = classify_image(uploaded_file.name)
            
            if label != "Error":
                #warna untuk progress bar
                primary_color = "#007BFF"  #biru untuk "Matang" || bisa diubah sesuai keinginan
                secondary_color = "#FF4136"  #merah untuk "Mentah" || bisa diubah sesuai keinginan
                label_color = primary_color if label == "Matang" else secondary_color
                
                #menampilkan hasil prediksi
                st.sidebar.write(f"**Nama File:** {uploaded_file.name}")
                st.sidebar.markdown(f"<h4 style='color: {label_color};'>Prediksi: {label}</h4>", unsafe_allow_html=True)
                
                #menampilkan confidence score
                st.sidebar.write("**Confidence:**")
                for i, class_name in enumerate(class_names):
                    st.sidebar.write(f"- {class_name}: {confidence[i] * 100:.2f}%")
                
                #menampilkan progress bar yang telah dibuat
                custom_progress_bar(confidence, primary_color, secondary_color)
                
                st.sidebar.write("---")
            else:
                st.sidebar.error(f"Kesalahan saat memproses gambar {uploaded_file.name}: {confidence}")
    else:
        st.sidebar.error("Silakan unggah setidaknya satu gambar untuk diprediksi.")

#tampilan di halaman utama untuk mengunggah dan menampilkan gambar yang diunggah
if uploaded_files:
    st.write("### Preview Gambar")
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"{uploaded_file.name}", use_column_width=True)
