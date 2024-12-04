import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

# Header aplikasi
st.title('Apple🍎 & Pear🍐Classification Using CNN Model Group 8')
st.markdown(
    """
    ### Selamat datang di Aplikasi Klasifikasi Buah Apel dan Pir Menggunakan Convolutional Neural Network!
    Unggah gambar buah apel atau pir, dan kami akan mengidentifikasinya untuk Anda. 🚀
    """
)

# Nama buah untuk klasifikasi
data_cat = ['Apel Biasa', 'Apel Hijau', 'Apel Merah', 'pir bulat', 'pir lonjong']

# Load model CNN
model = load_model('Image_classify.h5')

# Fungsi untuk klasifikasi gambar
def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = 'Gambar adalah ' + data_cat[np.argmax(result)] + ' dengan tingkat kemiripan ' + str(np.max(result) * 100) + '%'
    return outcome
    
# Buat folder upload jika belum ada
if not os.path.exists('upload'):
    os.makedirs('upload')

# Layout dengan Tabs
tab1, tab2 = st.tabs(["📂 Upload Gambar", "📊 Hasil Prediksi"])

with tab1:
    st.subheader("Unggah Gambar Anda di Sini")
    uploaded_file = st.file_uploader('Upload an Image')
    if uploaded_file is not None:
        # Simpan file yang diunggah ke folder upload
        file_path = os.path.join('upload', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Tampilkan gambar yang diunggah
        st.image(file_path, width=300, caption="Gambar yang diunggah", use_column_width="auto")

        # Prediksi hasil dan tampilkan di Tab Hasil
        with tab2:
            result = classify_images(file_path)
            st.markdown(f"### {result}")
