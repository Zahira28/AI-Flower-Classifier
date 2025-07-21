import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model (convert your TFJS model to Keras H5 or SavedModel beforehand)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("flowerclassifierai2.h5")
    return model

model = load_model()

class_names = [
    'Aster', 'Astilbe', 'Bellflower', 'Black Eyed Susan', 'Calendula', 'Carnation', 'Coreopsis', 'Daffodil', 'Daisy',
    'Dandelion', 'Iris', 'Lavender', 'Lily', 'Lotus', 'Magnolia', 'Marigold', 'Orchid', 'Poppy', 'Rose', 'Sunflower',
    'Tulip', 'Water Lily'
]

st.set_page_config(page_title="🌸 FloraLens - Flower Classifier AI", layout="centered")
st.title("🌸 FloraLens")
st.write("Kenali jenis bunga dengan AI yang cerdas! Upload gambar bunga di bawah ini.")

uploaded_file = st.file_uploader("Upload foto bunga", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Foto Bunga Diupload", use_column_width=True)

        # Preprocess image
        img = image.resize((299, 299))
        img_array = np.array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        with st.spinner("🔍 Sedang menganalisis bunga..."):
            prediction = model.predict(img_batch)
            probabilities = tf.nn.softmax(prediction[0])
            pred_index = tf.argmax(probabilities).numpy()
            confidence = probabilities[pred_index].numpy()

        # Show results
        st.success("🎉 Hasil Analisis")
        st.write(f"**Jenis bunga:** {class_names[pred_index]}")
        st.write(f"**Tingkat kepercayaan:** {confidence * 100:.2f}%")

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses gambar: {e}")
