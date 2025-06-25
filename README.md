# Laporan Proyek Machine Learning
### Muhammad Ilham Ramadhan - MC004D5Y2072

## Domain Proyek
<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/e/e8/Tesla_logo.png" alt="Tesla Logo" width="150"/>
</p>

Harga saham merupakan indikator utama yang mencerminkan kondisi pasar dan ekspektasi investor terhadap suatu perusahaan. Tesla Inc. (TSLA) sebagai salah satu perusahaan teknologi otomotif terbesar di dunia telah menarik perhatian investor global karena pertumbuhannya yang cepat dan volatilitas harga sahamnya yang tinggi. Menurut data dari Yahoo Finance, pergerakan harga saham Tesla selama beberapa tahun terakhir menunjukkan pola yang kompleks dan fluktuatif, yang dipengaruhi oleh berbagai faktor seperti laporan keuangan, inovasi teknologi, keputusan manajemen, hingga sentimen pasar global [[1]](https://finance.yahoo.com/quote/TSLA/). 

Prediksi harga saham yang akurat sangat penting untuk mendukung pengambilan keputusan investasi yang lebih tepat. Oleh karena itu, pemanfaatan metode deep learning seperti **Long Short-Term Memory (LSTM)** dan **Gated Recurrent Unit (GRU)** menjadi fokus utama dalam penelitian ini. Kedua arsitektur tersebut dirancang untuk menangani data deret waktu (time series) dan memiliki kemampuan dalam mengenali pola jangka panjang dalam data historis. Berbagai studi telah menunjukkan bahwa model LSTM dan GRU mampu memberikan hasil prediksi yang lebih stabil dan akurat dibandingkan metode tradisional dalam konteks pasar keuangan [[2]](https://doi.org/10.1016/j.ejor.2017.11.054)[[3]](https://doi.org/10.1016/j.asoc.2020.106181).

## Business Understanding

### Problem Statements
Berdasarkan uraian yang telah dipaparkan pada latar belakang di atas, maka dapat dirumuskan beberapa permasalahan sebagai berikut:

1. Bagaimana membangun model deep learning yang dapat memprediksi harga saham Tesla berdasarkan data historisnya?
2. Model deep learning mana yang memiliki performa prediksi terbaik antara LSTM dan GRU berdasarkan metrik evaluasi seperti MSE, MAE, RMSE, dan MAPE?

---

### Goals
Berdasarkan rumusan masalah di atas, maka tujuan dari proyek ini adalah sebagai berikut:

1. Membangun dua model deep learning, yaitu LSTM dan GRU, untuk melakukan prediksi harga saham Tesla menggunakan data time series historis.

2. Melakukan evaluasi dan perbandingan performa antara model LSTM dan GRU menggunakan metrik evaluasi Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan Mean Absolute Percentage Error (MAPE) untuk menentukan model yang paling optimal.

---

### Solution Statements
Berdasarkan tujuan yang telah ditetapkan, solusi yang diusulkan dalam proyek ini adalah sebagai berikut:

1. Mengimplementasikan dua model deep learning yaitu LSTM dan GRU untuk mempelajari pola harga saham Tesla dan menghasilkan prediksi harga penutupan (closing price) secara harian.

2. Melakukan evaluasi performa kedua model berdasarkan tiga metrik utama: MSE, MAE, RMSE, dan MAPE untuk menilai akurasi dan kestabilan hasil prediksi. Pemilihan model terbaik akan dilakukan berdasarkan nilai metrik terkecil.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **Tesla Stock Data** yang tersedia di [Kaggle](https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021). Dataset ini berisi data historis harga saham Tesla Inc. pada tanggal 29 Juni 2010 hingga 24 Maret 2022 dan sering digunakan untuk eksperimen time series forecasting maupun analisis pasar saham. Dataset mencerminkan pergerakan harga saham Tesla dari waktu ke waktu yang sangat berguna untuk membangun model prediksi berbasis deep learning.

### Informasi Dataset

| Jenis      | Keterangan                                                                                   |
|------------|----------------------------------------------------------------------------------------------|
| Title      | Tesla Stock Data                                                  |
| Source     | [Kaggle](https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021)  |
| Owner      | [Arpit Verma](https://www.kaggle.com/varpit94)                                                     |
| License    | Other                                                            |
| Visibility | Publik                                                                                       |
| Tags       | Finance, Investing, Time Series Analysis                                                |
| Usability  | 10.00                                                                                          |

### Variabel-variabel pada Dataset Saham Tesla

Dataset ini berisi **2956 entri** dan terdiri dari beberapa kolom utama yang merepresentasikan harga saham Tesla pada tiap harinya, termasuk volume perdagangan. Berikut adalah penjelasan masing-masing variabel.

- **Date** : Tanggal transaksi saham menjadi indeks utama dalam prediksi time series.

- **Open** : Harga pembukaan saham pada awal perdagangan hari tersebut.

- **High** : Harga tertinggi saham pada hari itu.

- **Low** : Harga terendah saham pada hari itu.

- **Close** : Harga penutupan saham pada akhir sesi perdagangan.

- **Adj Close** : Harga penutupan yang telah disesuaikan dengan dividen dan pemecahan saham.

- **Volume** : Jumlah saham Tesla yang diperdagangkan pada hari tersebut.

Dalam proyek ini, fokus utama prediksi adalah pada kolom **`Close`**, yaitu harga penutupan karena kolom ini paling sering digunakan dalam analisis tren dan pengambilan keputusan investasi.

### Exploratory Data Analysis - Deskripsi Variabel

Berdasarkan hasil eksplorasi awal terhadap dataset saham Tesla, diketahui informasi awal data sebagai berikut.

| Jumlah Baris | Jumlah Kolom |
| ------------ | ------------ |
| 2.956        | 6            |

Dataset ini memiliki 2.956 baris data dengan 6 fitur yang semuanya bertipe numerik, dan tidak terdapat missing value. Dataset ini juga sudah menggunakan indeks bertipe waktu (`DatetimeIndex`) yang mencakup periode dari tanggal 2010-06-29 hingga 2022-03-24.

| # | Column    | Non-Null Count | Dtype   |
| - | --------- | -------------- | ------- |
| 0 | Open      | 2.956 non-null | float64 |
| 1 | High      | 2.956 non-null | float64 |
| 2 | Low       | 2.956 non-null | float64 |
| 3 | Close     | 2.956 non-null | float64 |
| 4 | Adj Close | 2.956 non-null | float64 |
| 5 | Volume    | 2.956 non-null | int64   |

Seluruh fitur bersifat numerik dan memiliki jumlah data yang lengkap (non-null 100%). Fitur-fitur tersebut menjelaskan harga dan volume transaksi saham Tesla pada setiap tanggal.

|       | Open    | High    | Low     | Close   | Adj Close | Volume      |
| ----- | ------- | ------- | ------- | ------- | --------- | ----------- |
| Count | 2.956   | 2.956   | 2.956   | 2.956   | 2.956     | 2.956       |
| Mean  | 138.69  | 141.77  | 135.43  | 138.76  | 138.76    | 31.31 juta  |
| Std   | 250.04  | 255.86  | 243.77  | 250.12  | 250.12    | 27.98 juta  |
| Min   | 3.23    | 3.33    | 2.99    | 3.16    | 3.16      | 592.500     |
| 25%   | 19.63   | 20.40   | 19.13   | 19.62   | 19.62     | 13.10 juta  |
| 50%   | 46.66   | 47.49   | 45.82   | 46.55   | 46.55     | 24.89 juta  |
| 75%   | 68.06   | 69.36   | 66.91   | 68.10   | 68.10     | 39.74 juta  |
| Max   | 1234.41 | 1243.49 | 1217.00 | 1229.91 | 1229.91   | 304.69 juta |

Dari informasi statistik deskriptif tersebut dapat disimpulkan:

- Harga saham Tesla mengalami peningkatan yang sangat besar dari nilai minimum di sekitar $3 hingga mencapai maksimum lebih dari $1.200, menunjukkan fluktuasi harga yang ekstrem.

- Volume perdagangan juga sangat bervariasi, dari hanya 592.500 saham per hari hingga mencapai lebih dari 304 juta saham.

- Nilai rata-rata Volume sekitar 31 juta saham per hari, menandakan saham ini tergolong aktif diperdagangkan.

### Exploratory Data Analysis - Missing Value dan Duplikasi Data

#### **Penanganan Missing Value**

Langkah awal pada tahap eksplorasi data adalah memeriksa apakah terdapat nilai yang hilang (missing value) dalam dataset. Hal ini penting untuk memastikan kualitas data sebelum dilakukan pemodelan lebih lanjut.

Pengecekan missing value dilakukan menggunakan kode berikut.
```
# Cek missing value
print(f'Missing Value :\n{df.isnull().sum()}')
```
Hasil dari pengecekan menunjukkan banyaknya missing value pada seluruh fitur dataset sebagai berikut.

| Kolom     | Missing Value |
| --------- | ------------- |
| Open      | 0             |
| High      | 0             |
| Low       | 0             |
| Close     | 0             |
| Adj Close | 0             |
| Volume    | 0             |

Berdasarkan hasil tersebut, tidak ada missing value pada dataset sehingga tidak perlu dilakukan penanganan missing value imputasi maupun penghapusan baris.

#### **Penanganan Duplikasi Data**

Setelah memastikan tidak adanya missing value, langkah berikutnya adalah memeriksa apakah terdapat baris data yang duplikat. Duplikasi dapat terjadi karena kesalahan saat pengumpulan atau penggabungan data dan bisa memengaruhi hasil analisis.

Pengecekan duplikasi dilakukan dengan kode berikut.
```
# Cek duplikasi data
jumlah_duplikasi = df.duplicated().sum()
print(f"Jumlah duplikasi data: {jumlah_duplikasi}")
```
Berdasarkan hasil pengecekan duplikasi data, didapatkan bahwa tidak terdapat duplikat data pada dataset sehingga tidak diperlukan penghapusan data dan bisa dilanjutkan ke analisis selanjutnya.

### Exploratory Data Analysis - Data Visualization

#### **Volume Penjualan Saham dari Waktu ke Waktu**

Kode berikut digunakan untuk membuat visualisasi volume penjualan saham Tesla dari waktu ke waktu.
```
# Visualisasi volume penjualan
plt.figure(figsize=(10, 6))
df['Volume'].plot()
plt.ylabel('Volume')
plt.xlabel('Date')
plt.title("Sales Volume")
plt.tight_layout()
plt.show()
```
Hasil visualisasi dari kode tersebut adalah terbentuk grafik sebagai berikut.

![alt text](image-4.png) 
 
Visualisasi tersebut menunjukkan bahwa volume penjualan saham relatif stabil dari tahun 2010 hingga 2013, kemudian meningkat tajam pada periode 2013–2014 yang mengindikasikan meningkatnya minat pasar. Lonjakan paling signifikan terjadi pada tahun 2020, kemungkinan besar akibat peristiwa besar seperti pandemi yang memengaruhi sentimen pasar secara global. Setelah itu, volume cenderung fluktuatif namun dengan tren menurun hingga 2022. Pola ini menunjukkan bahwa volume perdagangan sangat dipengaruhi oleh peristiwa ekonomi makro dan perilaku investor terhadap ketidakpastian pasar.

#### **Harga Saham Harian Tertinggi dan Terendah**

Untuk memvisualisasikan harga saham harian tertinggi dan terendah gunakan kode berikut.
```
# Visualisasi harga saham harian tertinggi dan terendah
plt.figure(figsize=(10, 6))
df['Low'].plot()
df['High'].plot()
plt.ylabel('Price')
plt.xlabel('Date')
plt.title("Low & High Price")
plt.legend(['Low Price', 'High Price'])
plt.tight_layout()
plt.show()
```
Dengan menggunakan kode tersebut, didapatkan visualisasi sebagai berikut.

![alt text](image-5.png)

Harga tertinggi dan terendah harian menunjukkan tren yang stabil dan cukup sempit sebelum tahun 2020, dengan harga berada di bawah $100. Namun, mulai tahun 2020 terjadi lonjakan harga yang drastis, di mana harga mencapai puncak baru, menunjukkan fase pertumbuhan eksponensial saham. Pada saat yang sama, jarak antara harga tertinggi dan terendah harian juga melebar, mengindikasikan volatilitas pasar yang tinggi. Hal ini mencerminkan periode yang sangat aktif secara spekulatif, di mana harga bergerak cepat dalam satu hari perdagangan.

#### **Harga Saham Pembukaan dan Penutupan Harian**

Untuk melihat perubahan harga saham pembukaan dan penutupan harian, gunakan kode berikut.
```
# Visualisasi harga pembukaan dan penutupan saham
plt.figure(figsize=(10, 6))
df['Open'].plot()
df['Close'].plot()
plt.ylabel('Price')
plt.xlabel('Date')
plt.title("Opening & Closing Price")
plt.legend(['Open Price', 'Close Price'])
plt.tight_layout()
plt.show()
```

Dengan menggunakan kode tersebut, didapatkan visualisasi perbedaan harga pembukaan dan penutupan saham harian sebagai berikut.

![alt text](image-6.png)

Harga pembukaan dan penutupan saham sangat berkorelasi erat dan menunjukkan pergerakan yang hampir bersamaan sepanjang waktu. Sebelum tahun 2020, harga saham cenderung rendah dan stabil, namun setelah itu terjadi tren kenaikan yang tajam. Kedekatan nilai antara harga pembukaan dan penutupan mengindikasikan kestabilan harga dalam intraday trading, yang berarti tidak banyak fluktuasi dalam satu hari. Namun, tren kenaikan yang berkelanjutan menunjukkan adanya sentimen positif dan minat beli yang kuat dari investor, terutama selama masa pertumbuhan perusahaan atau peristiwa global tertentu.

#### **Candlestick Chart**

Untuk menampilkan pergerakan harga saham secara lebih detail dari waktu ke waktu, digunakan grafik candlestick yang menggambarkan harga pembukaan, penutupan, tertinggi, dan terendah dalam satu periode. Visualisasi ini dibuat dengan kode seperti berikut.
```
# Visualisasi candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close']
)])
fig.update_layout(title='Candlestick Chart', xaxis_rangeslider_visible=False)
fig.show()
```

Hasil dari kode tersebut adalah visualisasi seperti di bawah ini.

![alt text](newplot-1.png)

Grafik candlestick ini menunjukkan tren harga saham dari tahun 2011 hingga awal 2022. Terlihat bahwa harga saham mulai mengalami peningkatan yang signifikan sejak tahun 2020, dengan volatilitas yang juga semakin tinggi. Hal ini kemungkinan disebabkan oleh sentimen pasar yang kuat atau perubahan signifikan pada kondisi perusahaan atau pasar global.

#### **Hubungan antar Variabel Harga Saham dan Volume**

Untuk melihat korelasi dan distribusi antar variabel seperti Open, High, Low, Close, dan Volume, digunakan pairplot berikut. Kode yang digunakan untuk membuat visualisasi ini adalah sebagai berikut.
```
# Visualisai pairplot untuk melihat hubungan antar fitur
sns.pairplot(df[['Open', 'High', 'Low', 'Close', 'Volume']])
```

Berikut adalah hasil visualisasinya untuk kode tersebut.

![alt text](image-7.png)

Berdasarkan grafik di atas, tampak bahwa harga pembukaan, tertinggi, terendah, dan penutupan saham memiliki korelasi yang sangat kuat, yang ditunjukkan oleh pola diagonal yang sangat rapat. Sementara itu, volume transaksi memiliki hubungan yang lebih lemah terhadap harga, meskipun terlihat bahwa volume cenderung menurun saat harga meningkat drastis. Ini bisa mengindikasikan bahwa pergerakan harga yang besar sering terjadi dalam kondisi volume yang tidak selalu tinggi, atau adanya aksi spekulatif pada saat-saat tertentu.

## Data Preparation

Pada tahap ini dilakukan serangkaian langkah untuk mempersiapkan data sebelum dimasukkan ke dalam model. Data Preparation pada proyek ini terdiri dari 4 tahap utama, yaitu:
1) Feature Selection

2) Feature Scaling

3) Data Splitting

4) Sequence Generation

### **Feature Selection**

Langkah awal adalah memilih kolom fitur yang akan digunakan untuk prediksi. Karena fokus prediksi adalah harga penutupan (Close), maka hanya kolom tersebut yang dipilih dari dataset.
```
# Mengambil hanya kolom harga penutupan
df = df["Close"]
df = pd.DataFrame(df)

# Mengubah ke dalam format array
data = df.values
```
Dengan langkah ini, dataset difokuskan hanya pada variabel target yaitu harga penutupan saham Tesla.

### **Feature Scaling**

Karena model deep learning sensitif terhadap skala data, maka dilakukan normalisasi data menggunakan `MinMaxScaler`. Proses ini mengubah nilai menjadi rentang 0 hingga 1.
```
# Normalisasi data ke rentang 0-1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
```
Normalisasi membantu model untuk berkonvergensi lebih cepat dan menghasilkan prediksi yang lebih akurat.

### **Data Splitting**

Selanjutnya, dataset dibagi menjadi data latih dan data uji. Proporsi yang digunakan adalah 80% untuk training dan 20% untuk testing. Karena ini adalah data time series, pemisahan dilakukan tanpa pengacakan (non-random split) untuk menjaga urutan waktu.
```
# Menentukan panjang urutan data historis
sequence = 30

# Hitung ukuran data
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size

# Split data
train_data = scaled_data[:train_size, 0:1]
test_data = scaled_data[train_size-sequence:, 0:1]
```
Perlu diperhatikan bahwa `test_data` dimulai dari `train_size - sequence` untuk memastikan adanya overlap window sebagai konteks prediksi pada data uji.

### **Sequence Generation**

Model seperti LSTM/GRU membutuhkan input berbentuk urutan (sequence), bukan data statis. Oleh karena itu, dibuatlah sliding window sepanjang 30 data terakhir untuk memprediksi data berikutnya.

#### **Data Training**

```
# Membuat sliding window untuk data training
X_train = []
y_train = []

for i in range(sequence, len(train_data)):
    X_train.append(train_data[i-sequence:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
```

#### **Data Testing**

```
# Membuat sliding window untuk data testing
X_test = []
y_test = []

for i in range(sequence, len(test_data)):
    X_test.append(test_data[i-sequence:i, 0])
    y_test.append(test_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
```

#### **Hasil Bentuk Data**

```
# Cek ukuran data
X_train.shape , y_train.shape
X_test.shape , y_test.shape
```
Pada output ukuran `X_train` dan `X_test` memiliki bentuk 3 dimensi `(samples, timesteps, features)` yang sesuai dengan input untuk model sekuensial. Pada data Training didapatkan ukuran `X_train` dan `y_train` dengan bentuk `((2334, 30, 1), (2334,))`, sementara pada Data Testing `X_test` dan `y_test` berukuran `((592, 30, 1), (592,))`

### Mengapa Data Preparation perlu dilakukan?

1) Feature Selection membantu menyederhanakan fokus model pada target prediksi yaitu harga penutupan.

2) Feature Scaling penting agar semua nilai berada dalam skala yang sama, menghindari dominasi fitur tertentu.

3) Data Splitting memungkinkan evaluasi performa model pada data yang belum pernah dilihat.

4) Sequence Generation memungkinkan model memahami pola historis dalam data time series.

## Modeling

Pada tahap ini, dilakukan pemodelan prediksi harga saham menggunakan dua pendekatan berbasis Recurrent Neural Network (RNN), yaitu Long Short-Term Memory (LSTM) dan Gated Recurrent Unit (GRU). Kedua model ini sangat cocok digunakan untuk data time series karena mampu menangkap hubungan jangka panjang dalam data sekuensial seperti harga saham. Pemodelan dilakukan dalam dua tahap utama sebagai berikut.

#### **Baseline Modeling**
Membangun model awal tanpa tuning hiperparameter secara eksplisit. Ini bertujuan untuk mengevaluasi performa dasar dari masing-masing arsitektur model terhadap data pelatihan.

#### **Early Stopping**
Diterapkan untuk menghindari overfitting. Model akan berhenti dilatih jika tidak ada peningkatan pada validation loss selama beberapa epoch berturut-turut.

Setiap model dilatih pada data historis saham Tesla dan divalidasi menggunakan data validasi untuk mengukur kemampuannya dalam melakukan generalisasi.

### Long Short-Term Memory (LSTM)

LSTM adalah salah satu arsitektur RNN yang memiliki kemampuan untuk mengingat informasi dalam jangka panjang dan mengatasi masalah vanishing gradient. Arsitektur ini banyak digunakan untuk pemodelan deret waktu (time series), seperti prediksi harga saham.

#### **Arsitektur Model**
Model LSTM yang dibangun terdiri dari beberapa lapisan sebagai berikut.
```
# Arsitektur model LSTM
model_lstm = Sequential()
model_lstm.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(LSTM(32, return_sequences=False))
model_lstm.add(Dense(16))
model_lstm.add(Dense(1))

model_lstm.summary()
```
Model terdiri dari:

- LSTM(64)  
Layer LSTM pertama dengan 64 unit, mengembalikan seluruh urutan (return_sequences=True).

- LSTM(32)  
Layer LSTM kedua dengan 32 unit, hanya mengembalikan output terakhir.

- Dense(16)  
Fully connected layer dengan 16 unit sebagai intermediate layer.

- Dense(1)  
Output layer dengan 1 unit untuk prediksi nilai harga saham (regresi).

Setelah model dibuat, kita dapat melihat ringkasan struktur model LSTM dengan perintah `model_lstm.summary()`.

| Layer (type)     | Output Shape   | Param #    |
| ---------------- | -------------- | ---------- |
| lstm (LSTM)      | (None, 30, 64) |     16,896 |
| lstm_1 (LSTM)    | (None, 32)     |     12,416 |
| dense (Dense)    | (None, 16)     |        528 |
| dense_1 (Dense)  | (None, 1)      |         17 |
  

Total params: 29,857  
Trainable params: 29,857  
Non-trainable params: 0

#### **Kompilasi dan Pelatihan Model**

Setelah model LSTM selesai dibangun, langkah selanjutnya adalah kompilasi dan pelatihan (training). Proses ini sangat penting karena menentukan bagaimana model belajar dari data dan mengevaluasi performanya.

- Kompilasi model bertujuan untuk mengatur fungsi loss (kerugian), algoritma optimisasi, dan metrik evaluasi.

- Training model adalah proses di mana model mempelajari pola dari data pelatihan selama beberapa epoch (putaran), untuk meminimalkan nilai loss.

Berikut adalah syntax yang digunakan untuk melakukan kompilasi dan training model LSTM.
```
# Kompilasi model
model_lstm.compile(
    loss='mse',
    optimizer='Adam',
    metrics=['mae']
)
```
- `loss='mse`'  
Fungsi loss yang digunakan adalah Mean Squared Error (MSE), yaitu selisih kuadrat antara nilai aktual dan nilai prediksi. MSE cocok untuk masalah regresi seperti prediksi harga saham.

- `optimizer='Adam'`  
Digunakan algoritma Adam (Adaptive Moment Estimation) yang sangat populer dalam deep learning karena mampu menyesuaikan laju pembelajaran secara adaptif untuk setiap parameter.

- `metrics=['mae']`  
Selain fungsi loss, digunakan metrik tambahan Mean Absolute Error (MAE) untuk memantau rata-rata kesalahan absolut prediksi terhadap nilai sebenarnya.

Selanjutnya, digunakan teknik Early Stopping untuk menghentikan pelatihan jika model sudah tidak mengalami perbaikan pada data validasi. Ini sangat berguna untuk menghindari overfitting, yaitu kondisi ketika model terlalu cocok terhadap data latih dan tidak mampu generalisasi ke data baru.
```
# Pelatihan model dengan Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```
- `monitor='val_loss'`  
Model akan memantau nilai kerugian (loss) pada data validasi.

- `patience=5`  
Jika tidak ada penurunan val_loss selama 5 epoch berturut-turut, pelatihan akan dihentikan.

- `restore_best_weights=True`  
Model akan mengembalikan bobot terbaik yang diperoleh selama pelatihan (bukan bobot di epoch terakhir).

Tahapan terakhir adalah model dilatih menggunakan data training. Berikut adalah kode untuk melatih data training dengan menggunakan model yang sudah didefinisikan.
```
# Melatih model
history_lstm = model_lstm.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping], 
    verbose=1
)
```
- `X_train, y_train`  
Data pelatihan fitur dan target.

- `epochs=50`  
Maksimal pelatihan dilakukan selama 50 epoch.

- `batch_size=32`  
Data dibagi menjadi batch berisi 32 data setiap kali update bobot.

- `validation_split=0.1`  
Sebanyak 10% data pelatihan digunakan sebagai data validasi.

- `callbacks=[early_stopping]`  
Memasukkan mekanisme early stopping dalam proses training.

- `verbose=1`  
Menampilkan log pelatihan secara rinci setiap epoch.

Dengan proses ini, model LSTM dapat belajar dari data historis saham dan siap untuk digunakan dalam tahap evaluasi maupun prediksi.

### Gated Recurrent Unit (GRU)

Setelah membangun model LSTM, kita juga mencoba pendekatan lain yaitu menggunakan GRU (Gated Recurrent Unit). GRU merupakan varian dari arsitektur Recurrent Neural Network (RNN) yang mirip dengan LSTM namun lebih sederhana secara struktur, sehingga sering kali lebih cepat dilatih dan membutuhkan sumber daya komputasi lebih sedikit.

#### **Arsitektur Model**

Model GRU yang digunakan memiliki struktur berlapis untuk menangkap kompleksitas pola waktu dari data historis harga saham. Berikut adalah susunan arsitektur model.
```
# Arsitektur model GRU
model_gru = Sequential()
model_gru.add(GRU(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_gru.add(GRU(32, return_sequences=False))
model_gru.add(Dense(16))
model_gru.add(Dense(1))
```
- `GRU(64, return_sequences=True)`  
Lapisan pertama GRU dengan 64 unit digunakan untuk memproses urutan input. return_sequences=True memastikan output berbentuk urutan untuk diteruskan ke lapisan GRU berikutnya.

- `GRU(32, return_sequences=False)`  
Lapisan GRU kedua dengan 32 unit hanya mengeluarkan output terakhir dari urutan, sebagai representasi ringkasan fitur waktu.

- `Dense(16)`  
Lapisan fully connected dengan 16 neuron untuk memproses hasil dari GRU dan membentuk representasi non-linier.

- `Dense(1)`   
Lapisan output untuk menghasilkan satu nilai prediksi harga penutupan.

Setelah model dibuat, kita dapat melihat ringkasan struktur model GRU dengan perintah model_gru.summary(). Output-nya mirip dengan LSTM namun dengan jumlah parameter yang sedikit berbeda, mengingat struktur internal GRU lebih ringan daripada LSTM.

| Layer (type)     | Output Shape   | Param #    |
| ---------------- | -------------- | ---------- |
| gru (GRU)        | (None, 30, 64) |     12,864 |
| gru_1 (GRU)      | (None, 32)     |      9,408 |
| dense_2 (Dense)    | (None, 16)     |        528 |
| dense_3 (Dense)  | (None, 1)      |         17 |
  

Total params: 22,817  
Trainable params: 22,817  
Non-trainable params: 0

#### **Kompilasi dan Pelatihan Model**

Sama seperti pada LSTM, model GRU juga dikompilasi dan dilatih menggunakan pendekatan serupa. Kita tetap menggunakan fungsi loss `MSE`, optimizer `Adam`, dan metrik evaluasi `MAE`.
```
# Kompilasi model
model_gru.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae']
)
```
Kemudian, strategi early stopping juga diterapkan untuk menghentikan pelatihan ketika tidak ada perbaikan lebih lanjut pada data validasi.
```
# Pelatihan model dengan Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```
Model kemudian dilatih dengan memanfaatkan data pelatihan yang telah dibagi pada proses data splitting.
```
# Melatih model
history_gru = model_gru.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping], 
    verbose=1
)
```
GRU sering kali dipilih ketika ingin mendapatkan hasil yang kompetitif namun dengan waktu pelatihan yang lebih cepat dibandingkan LSTM. Struktur GRU lebih ringkas karena hanya memiliki dua gerbang (update dan reset), sementara LSTM memiliki tiga gerbang (input, forget, output), sehingga GRU lebih efisien secara komputasi.

## Evaluation

Pada tahap ini, dilakukan evaluasi terhadap performa model LSTM dan GRU untuk memprediksi harga penutupan saham Tesla. Evaluasi yang dilakukan mencakup perhitungan Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), dan Mean Absolute Percentage Error (MAPE). Berikut adalah definisi dan rumus dari masing-masing perhitungan metrik evaluasi yang digunakan.

- **Mean Squared Error (MSE)**  
Mengukur rata-rata kuadrat dari selisih antara nilai aktual dan nilai prediksi.

  $MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

- **Mean Absolute Error (MAE)**  
Mengukur rata-rata dari nilai absolut selisih antara nilai aktual dan prediksi.

  $MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$

- **Root Mean Squared Error (RMSE)**  
Akar dari MSE, memberikan penalti lebih besar untuk kesalahan yang lebih besar.

  $RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$

- **Mean Absolute Percentage Error (MAPE)**  
Mengukur kesalahan dalam bentuk persentase dari nilai aktual.

  $MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|$

---

### Evaluasi Model LSTM

#### Visualisasi Loss LSTM

Untuk memahami bagaimana model belajar selama proses pelatihan, dilakukan visualisasi terhadap nilai loss pada data training dan validation. Visualisasi ini berguna untuk mendeteksi overfitting atau underfitting.

```
plt.figure(figsize=(10, 6))
plt.plot(history_lstm.history['loss'], label='Training Loss')
plt.plot(history_lstm.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
```
![alt text](image.png)

Visualisasi ini menunjukkan bahwa model LSTM memiliki tren loss yang menurun dan stabil, menandakan proses pelatihan berjalan dengan baik tanpa indikasi overfitting yang signifikan.

#### Evaluasi Metrik LSTM

Untuk mengetahui seberapa baik model memprediksi data, digunakan beberapa metrik evaluasi. Berikut kode yang digunakan untuk menghitung MSE, MAE, RMSE, dan MAPE.
```
lstm_pred = model_lstm.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred)
lstm_test = scaler.inverse_transform(y_test.reshape(-1, 1))

print("Mean Square Error :", round(mean_squared_error(lstm_test, lstm_pred), 2))
print("Mean Absolute Error :", round(mean_absolute_error(lstm_test, lstm_pred), 2))
print("Root Mean Square Error :", np.sqrt(np.mean((lstm_test - lstm_pred)**2)).round(2))
print("Mean Absolute Percentage Error :", round(mean_absolute_percentage_error(lstm_test, lstm_pred), 2))
```
Dari kode tersebut, didapatkan hasil evaluasi metrik sebagai berikut.

- Mean Square Error : **3840.42**

MSE mengukur rata-rata dari kuadrat selisih antara nilai aktual dan nilai prediksi. Nilai sebesar 3840.42 menunjukkan bahwa terdapat beberapa prediksi yang meleset cukup jauh dari nilai sebenarnya, karena MSE sensitif terhadap outlier. Semakin besar nilai ini, semakin besar pula deviasi ekstrem yang terjadi.

- Mean Absolute Error : **45.48**

MAE mengukur rata-rata dari kesalahan absolut antara hasil prediksi dan data aktual. Dengan nilai MAE sebesar 45.48, dapat disimpulkan bahwa secara rata-rata model LSTM memiliki deviasi sekitar ±45.48 unit dari nilai sebenarnya. Ini memberikan gambaran langsung dan intuitif mengenai seberapa jauh kesalahan prediksi secara umum.

- Root Mean Square Error : **43.4**

RMSE adalah akar kuadrat dari MSE dan memiliki satuan yang sama dengan target data. Nilai RMSE sebesar 43.4 menunjukkan bahwa deviasi prediksi dari nilai aktual berada pada kisaran tersebut. RMSE yang tinggi mengindikasikan bahwa model mengalami kesulitan dalam mengikuti pola data secara akurat, terutama pada nilai-nilai yang berubah drastis.

- Mean Absolute Percentage Error : **0.07**

MAPE menyajikan kesalahan dalam bentuk persentase terhadap nilai aktual. Nilai MAPE sebesar 7% menandakan bahwa rata-rata kesalahan prediksi model terhadap nilai aktual berada pada angka 7%. Ini masih tergolong cukup baik untuk kasus prediksi harga, meskipun mungkin belum tergolong sangat presisi.

#### Visualisasi Hasil Prediksi LSTM

Untuk membandingkan hasil prediksi dengan data aktual, digunakan visualisasi grafik sebagai berikut.
```
train = df.iloc[:train_size , 0:1]
test = df.iloc[train_size: , 0:1]
test['LSTM Prediction'] = lstm_pred

plt.figure(figsize= (10, 6))
plt.title('Tesla Close Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(test['Close'])
plt.plot(test['LSTM Prediction'])
plt.legend(['Train', 'Test', 'LSTM Prediction'])
```
![alt text](image-1.png)

Visualisasi ini memperlihatkan bahwa model LSTM mampu mengikuti pola tren harga saham dengan cukup baik, meskipun terdapat beberapa deviasi pada nilai-nilai ekstrem.

---

### Evaluasi Model GRU

#### Visualisasi Loss GRU

Langkah pertama adalah memvisualisasikan tren loss selama pelatihan data training dan validation untuk model GRU. Berikut kode yang digunakan.
```
plt.figure(figsize=(10, 6))
plt.plot(history_gru.history['loss'], label='Training Loss')
plt.plot(history_gru.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
```
![alt text](image-2.png)

Grafik menunjukkan kurva loss yang menurun dan konvergen antara data training dan validation, menandakan bahwa model GRU stabil dan tidak mengalami overfitting.

#### Evaluasi Metrik GRU

Langkah selanjutnya adalah menghitung metrik performa dari model GRU, menggunakan kode berikut.

```
gru_pred = model_gru.predict(X_test)
gru_pred = scaler.inverse_transform(gru_pred)
gru_test = scaler.inverse_transform(y_test.reshape(-1, 1))

print("MSE:", round(mean_squared_error(gru_test, gru_pred), 2))
print("MAE:", round(mean_absolute_error(gru_test, gru_pred), 2))
print("Root Mean Square Error :", np.sqrt(np.mean((gru_test - gru_pred)**2)).round(2))
print("Mean Absolute Percentage Error :", round(mean_absolute_percentage_error(gru_test, gru_pred), 2))
```

Dari kode tersebut diperoleh hasil evaluasi metrik sebagai berikut.

- Mean Square Error : **908.78**

MSE mengukur rata-rata kuadrat dari selisih antara nilai aktual dan prediksi. Nilai ini cukup besar, yang menunjukkan bahwa terdapat deviasi yang cukup signifikan dalam beberapa prediksi. Namun, MSE sangat sensitif terhadap outlier sehingga nilai ini bisa terpengaruh oleh prediksi yang jauh dari nilai sebenarnya.

- Mean Absolute Error : **20.05**

MAE memberikan nilai rata-rata dari kesalahan absolut antara prediksi dan data aktual. Nilai MAE sebesar 20.05 berarti secara rata-rata model membuat kesalahan sebesar ±20 unit dari harga aktual. Nilai ini cukup informatif untuk mengetahui seberapa besar deviasi model secara umum tanpa dipengaruhi oleh outlier secara ekstrem.

- Root Mean Square Error : **5.44**

RMSE adalah akar dari MSE dan memiliki satuan yang sama dengan data asli. Nilai RMSE sebesar 5.44 mengindikasikan bahwa deviasi rata-rata model terhadap nilai aktual adalah sekitar 5.44 unit. Ini memberi gambaran yang cukup akurat tentang seberapa baik model mampu menangkap pola dalam data.

- Mean Absolute Percentage Error : **0.04**

MAPE mengukur kesalahan dalam bentuk persentase dari nilai aktual. Nilai MAPE sebesar 0.04 atau 4% menunjukkan bahwa model GRU mampu melakukan prediksi dengan tingkat kesalahan rata-rata sebesar 4% dari nilai aktual. Ini merupakan nilai yang sangat baik dalam konteks pemodelan time series harga saham, karena menunjukkan bahwa model cukup presisi dan andal.

#### Visualisasi Hasil Prediksi GRU

Visualisasi hasil prediksi GRU dibandingkan dengan data aktual dilakukan sebagai berikut.
```
train = df.iloc[:train_size , 0:1]
test = df.iloc[train_size: , 0:1]
test['GRU Prediction'] = gru_pred

plt.figure(figsize= (10, 6))
plt.title('Tesla Close Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(test['Close'])
plt.plot(test['GRU Prediction'])
plt.legend(['Train', 'Test', 'GRU Prediction'])
```
![alt text](image-3.png)

Grafik menunjukkan bahwa model GRU berhasil mengikuti pola tren dari data aktual dengan sangat baik, termasuk menangkap pergerakan harga yang signifikan.

---

## Kesimpulan

Setelah dilakukan evaluasi, berikut adalah perbandingan metrik dari kedua model:

| Model | MSE         | MAE         | RMSE        | MAPE        |
| ----- | ----------- | ----------- | ----------- | ----------- |
| LSTM  | 3840.42     | 45.48       | 43.4        | 0.07        |
| GRU   | 908.78      | 20.05       | 5.44        | 0.04        |

Melalui perbandingan metrik evaluasi antara hasil permodelan dengan LSTM dan GRU dapat disimpulkan bahwa Model GRU menghasilkan performa yang lebih baik dibandingkan LSTM berdasarkan semua metrik evaluasi.

Nilai MSE dan MAE GRU lebih rendah, yang berarti kesalahan prediksinya lebih kecil secara absolut maupun kuadrat. MAPE GRU sebesar 4% mengindikasikan bahwa rata-rata prediksi hanya meleset sebesar 4% dari nilai aktual, dibandingkan 7% pada LSTM. RMSE GRU juga menunjukkan bahwa sebaran kesalahan lebih kecil dibandingkan LSTM.

Dengan mempertimbangkan semua hasil di atas, model GRU lebih unggul dalam memprediksi harga saham Tesla dan lebih disarankan untuk digunakan dalam implementasi lebih lanjut. Performa yang baik ini juga membuka kemungkinan untuk meningkatkan akurasi dengan tuning tambahan atau penggabungan metode lain (ensemble, hybrid, dsb.).

## Referensi
[1] Yahoo Finance, “Tesla, Inc. (TSLA) Stock Price, News, Quote & History,” Yahoo Finance, 2025. [Online]. Available: https://finance.yahoo.com/quote/TSLA/. [Accessed: 26-May-2025].

[2] A. Fischer and C. Krauss, “Deep learning with long short-term memory networks for financial market predictions,” European Journal of Operational Research, vol. 270, no. 2, pp. 654–669, 2018, doi: 10.1016/j.ejor.2017.11.054.

[3] M. T. Nguyen, N. D. Vo, T. V. Nguyen, and Q. V. Pham, “Time series forecasting using GRU neural networks and model comparison with LSTM,” Applied Soft Computing, vol. 93, 106181, 2020, doi: 10.1016/j.asoc.2020.106181.