# utsML-DL

**Hands-On End-to-End Models (Classification, Regression & Clustering)**

## 1. Tujuan Repository

Repository ini berisi pengumpulan individu saya untuk **Machine Learning Midterm (UTS)** dengan tema:

> “Hands-On End-to-End Models Machine Learning and Deep Learning”

Fokus dalam repository ini adalah pada **machine learning tradisional / klasik** untuk:
- Fraud detection (binary classification)
- Song year prediction (regression)
- Customer segmentation (clustering)

Setiap task diimplementasikan sebagai **pipeline ML end-to-end**: mulai dari data loading, preprocessing, modeling, evaluation, hingga interpretasi.

---

## 2. Gambaran Proyek

### 2.1 Tujuan

- Membangun **pipeline machine learning end-to-end** untuk:
  1. **Fraud detection** pada transaksi online  
  2. **Regresi** untuk memprediksi tahun rilis lagu  
  3. **Customer clustering** berdasarkan perilaku kartu kredit
- Praktik yang dilakukan:
  - Data cleaning & preprocessing  
  - Menangani missing values dan outliers  
  - Menangani **class imbalance** (khususnya untuk fraud detection)  
  - Feature engineering / feature selection  
  - Training dan evaluasi beberapa model ML  
  - Melakukan **hyperparameter tuning dasar**  
  - Membandingkan performa model dan menginterpretasikan hasil

### 2.2 Task yang Diimplementasikan

1. **Fraud Detection – Binary Classification**  
   - Memprediksi probabilitas bahwa sebuah transaksi merupakan fraud (`isFraud = 1`).  
   - Menggunakan fitur transaksi seperti amount, waktu, product code, informasi kartu, alamat, dll.  
   - Output dapat berupa:
     - Performa model pada validation set  
     - Prediksi probabilitas fraud pada `test_transaction.csv` (format: `TransactionID, isFraud`)  

2. **Song Year Prediction – Regression**  
   - Memprediksi **tahun rilis lagu** berdasarkan fitur numerik audio.  
   - Kolom pertama = target (tahun rilis), sisanya = fitur numerik anonim (`feature_1`, `feature_2`, dst.).  

3. **Customer Clustering – Unsupervised Learning**  
   - Mengelompokkan pelanggan berdasarkan pola penggunaan kartu kredit.  
   - Menggunakan fitur seperti balance, purchases, cash advance, frekuensi transaksi, credit limit, payments, minimum payments, tenure, dll.  
   - Hasil cluster dapat diinterpretasikan sebagai: high spender, low spender, installment user, risky customer, dan lain-lain.  

---

## 3. Dataset

Semua dataset disediakan oleh dosen sebagai bagian dari UTS.

### 3.1 Fraud Detection – Transaction Data

- **`train_transaction.csv`**  
  - Berisi transaksi berlabel untuk training dan evaluasi.  
  - Setiap baris = satu transaksi online dengan banyak fitur.  
  - Kolom target: **`isFraud`**
    - `1` → transaksi fraud  
    - `0` → bukan fraud  

- **`test_transaction.csv`**  
  - Memiliki fitur yang sama seperti train namun tanpa `isFraud`.  
  - Digunakan untuk menghasilkan prediksi probabilitas fraud.  
  - Format output umum:
    - `TransactionID, isFraud`

---

### 3.2 Regression – Song Year Prediction

- **`midterm-regresi-dataset.csv`**  
  - Nilai pertama = target (tahun rilis).  
  - Nilai berikutnya = fitur numerik anonim hasil ekstraksi sinyal audio (seperti spektral, timbre, dll.)

---

### 3.3 Customer Clustering – Credit Card Dataset

- **`clusteringmidterm.csv`**  
  - Setiap baris = satu pelanggan kartu kredit.  
  - Kolom penting termasuk:
    - `CUST_ID`  
    - `BALANCE`, `BALANCE_FREQUENCY`  
    - `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`  
    - `PURCHASES_FREQUENCY`, `ONEOFF_PURCHASES_FREQUENCY`, `PURCHASES_INSTALLMENTS_FREQUENCY`  
    - `CASH_ADVANCE`, `CASH_ADVANCE_FREQUENCY`, `CASH_ADVANCE_TRX`  
    - `PURCHASES_TRX`  
    - `CREDIT_LIMIT`  
    - `PAYMENTS`, `MINIMUM_PAYMENTS`  
    - `PRC_FULL_PAYMENT`  
    - `TENURE`  

---

## 4. Struktur Proyek

> Catatan: sesuaikan nama file jika struktur sebenarnya berbeda.

```text
midterm-machine-learning/
├── data/
│   ├── train_transaction.csv
│   ├── test_transaction.csv
│   ├── midterm-regresi-dataset.csv
│   └── clusteringmidterm.csv
├── notebooks/
│   ├── 01_fraud_detection_classification_ml.ipynb
│   ├── 02_song_year_regression_ml.ipynb
│   └── 03_customer_clustering_ml.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models_classification.py
│   ├── models_regression.py
│   └── models_clustering.py
├── requirements.txt
└── README.md

