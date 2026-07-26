# 📰 Indonesian News Classification using Word2Vec and LSTM

## 📌 Project Overview

Project ini merupakan implementasi model Deep Learning untuk melakukan klasifikasi kategori berita berbahasa Indonesia menggunakan pendekatan **Word2Vec sebagai word embedding** dan **Long Short-Term Memory (LSTM)** sebagai model klasifikasi.

Model dikembangkan untuk mengklasifikasikan artikel berita ke dalam beberapa kategori berdasarkan pola dan representasi semantik dari teks.

---

## 🎯 Background & Objectives

Perkembangan jumlah berita digital yang semakin meningkat menyebabkan kebutuhan terhadap sistem otomatis untuk mengelompokkan berita menjadi semakin penting.

Klasifikasi berita secara manual membutuhkan waktu dan sumber daya yang besar. Oleh karena itu, project ini bertujuan untuk membangun model klasifikasi teks berbasis Deep Learning yang mampu mengidentifikasi kategori berita Bahasa Indonesia secara otomatis.

Tujuan project:

- Melakukan preprocessing terhadap teks berita Bahasa Indonesia.
- Membuat representasi kata menggunakan Word2Vec Skip-gram.
- Mengembangkan model klasifikasi menggunakan LSTM.
- Mengevaluasi performa model menggunakan K-Fold Cross Validation.

---

# 🗂 Dataset

Dataset yang digunakan berupa kumpulan artikel berita Bahasa Indonesia dengan total:

- Jumlah data: **150.466 artikel berita**
- Jumlah kategori: **8 kategori**

Kategori berita:

- Bisnis Ekonomi
- Bola
- Lifestyle
- Nasional
- Olahraga
- Otomotif
- Teknologi
- Travel

---

# 🔄 Data Preprocessing

Tahapan preprocessing yang dilakukan:

1. **Cleaning**
   - Menghapus karakter yang tidak diperlukan.
   - Menghapus URL dan simbol.

2. **Case Folding**
   - Mengubah seluruh teks menjadi huruf kecil.

3. **Tokenization**
   - Memecah teks menjadi token kata.

4. **Stopword Removal**
   - Menghapus kata umum menggunakan library Sastrawi.

5. **Padding**
   - Menyamakan panjang sequence menggunakan nilai maksimum 200 token.

---

# 🧠 Model Architecture

Model menggunakan kombinasi:
Input Text
|
↓
Preprocessing
|
↓
Tokenizer
|
↓
Word2Vec Embedding Layer
|
↓
LSTM (128 Units)
|
↓
Dense Layer (32 Neuron)
|
↓
Softmax Output
|
↓
News Category

---

# 🔤 Word Embedding

Metode embedding yang digunakan:

- Algorithm: Word2Vec
- Training Method: Skip-gram
- Vector Dimension: 100
- Window Size: 5
- Epoch: 10

Word2Vec digunakan untuk menghasilkan representasi numerik kata berdasarkan hubungan konteks antar kata dalam artikel berita.

---

# ⚙️ Experiment Setup

Eksperimen dilakukan menggunakan:

| Parameter           | Nilai             |
| ------------------- | ----------------- |
| Model               | LSTM              |
| Optimizer           | Adam              |
| Learning Rate       | 0.001             |
| Batch Size          | 32                |
| Epoch Maximum       | 50                |
| Validation          | 10% training data |
| Cross Validation    | 10-Fold K-Fold    |
| Max Sequence Length | 200               |

---

# 📊 Results

Evaluasi dilakukan menggunakan:

- Accuracy
- Precision
- Recall
- F1-Score

Hasil terbaik:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 92.03% |
| Precision | 92.07% |
| Recall    | 92.03% |
| F1-Score  | 92.03% |

---

# 🚀 Deployment

Model telah diimplementasikan menggunakan **Streamlit** sebagai aplikasi web sederhana.

Fitur aplikasi:

- Input teks berita.
- Melakukan preprocessing otomatis.
- Prediksi kategori berita.
- Menampilkan confidence score.
- Menampilkan probabilitas setiap kategori.

Menjalankan aplikasi:

```bash
streamlit run app.py
```
