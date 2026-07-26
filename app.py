import streamlit as st
import numpy as np
import pickle
import re
import json

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =====================================================
# KONFIGURASI
# =====================================================

WEIGHTS_PATH = "models/model_final_weights.weights.h5"
CONFIG_PATH = "models/model_config.pkl"
EMBEDDING_MATRIX_PATH = "models/embedding_matrix_final.npy"
TOKENIZER_PATH = "models/tokenizer_final_deploy.pkl"
LABEL_ENCODER_PATH = "models/label_encoder_deploy.pkl"

# =====================================================
# BANGUN ULANG ARSITEKTUR (HARUS SAMA PERSIS DENGAN SAAT TRAINING)
# =====================================================

def create_model(vocab_size, embedding_matrix, embedding_dim, num_classes):
    model = Sequential()
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False
    ))
    model.add(LSTM(128))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_artifacts():

    with open(CONFIG_PATH, "rb") as f:
        config = pickle.load(f)

    embedding_matrix = np.load(EMBEDDING_MATRIX_PATH)

    model = create_model(
        vocab_size=config["vocab_size"],
        embedding_matrix=embedding_matrix,
        embedding_dim=config["embedding_dim"],
        num_classes=config["num_classes"]
    )

    model.build(input_shape=(None, config["max_len"]))

    model.load_weights(WEIGHTS_PATH)

    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder, config


model, tokenizer, label_encoder, config = load_artifacts()
MAX_LEN = config["max_len"]

# =====================================================
# PREPROCESSING
# =====================================================

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

LABEL_MAPPING_PATH = "models/labelmapping.json"

with open(LABEL_MAPPING_PATH, 'r', encoding='utf-8') as f:
    _raw_mapping = json.load(f)

label_mapping = {v: k for k, v in _raw_mapping.items()}

def preprocess_text(text):
    """
    Preprocessing sesuai proses training:
    - Lowercase
    - Hapus URL
    - Hapus angka
    - Hapus tanda baca/simbol
    - Rapikan spasi
    - Stopword Removal (Sastrawi)
    """

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = stopword.remove(text)

    return text


# =====================================================
# PREDIKSI
# =====================================================

def predict_category(text):
    processed = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post")

    prediction = model.predict(padded, verbose=0)
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    predicted_label = label_mapping[predicted_index]

    return predicted_label, confidence, prediction[0]


# =====================================================
# HALAMAN
# =====================================================

st.set_page_config(
    page_title="Indonesian News Classification",
    page_icon="📰",
    layout="wide"
)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.title("📚 Informasi Model")
    st.markdown("### Model")
    st.write("**LSTM**")
    st.markdown("### Word Embedding")
    st.write("Word2Vec (Skip-gram)")
    st.markdown("### Dataset")
    st.write("150.466 Artikel Berita")
    st.markdown("### Jumlah Kategori")
    st.write("8 Kategori")
    st.markdown("### Evaluasi")
    st.write("10-Fold Cross Validation")
    st.markdown("### Akurasi")
    st.success("92.03 %")

# =====================================================
# HEADER
# =====================================================

st.title("📰 Indonesian News Classification")

st.write(
    """
Aplikasi ini mengklasifikasikan artikel berita Bahasa Indonesia
menggunakan model **Word2Vec + LSTM**.

Masukkan isi berita pada kolom di bawah, kemudian tekan tombol
**Prediksi** untuk mengetahui kategori berita.
"""
)

# =====================================================
# INPUT
# =====================================================

input_text = st.text_area(
    "Masukkan Isi Berita",
    height=250,
    placeholder="Tempel artikel berita di sini..."
)

# =====================================================
# PREDIKSI
# =====================================================

if st.button("🔍 Prediksi"):

    if input_text.strip() == "":
        st.warning("Silakan masukkan teks berita terlebih dahulu.")
    else:
        with st.spinner("Sedang melakukan prediksi..."):
            label, confidence, probabilities = predict_category(input_text)

        st.success(f"Kategori Berita : **{label}**")
        st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
        st.divider()
        st.subheader("Probabilitas Tiap Kategori")

        for idx, prob in enumerate(probabilities):
            cls = label_mapping[idx]
            st.write(f"**{cls}**")
            st.progress(float(prob))
            st.caption(f"{prob*100:.2f}%")

        st.divider()
        with st.expander("Lihat Hasil Preprocessing"):
            st.write(preprocess_text(input_text))