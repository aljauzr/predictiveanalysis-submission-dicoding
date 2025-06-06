# -*- coding: utf-8 -*-
"""notebook.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dnqTEaMzTUECDKnMWLtU7wAmfQKKUqO6

Import Library
---
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns
from sklearn.datasets import fetch_california_housing

"""Data Understanding
---

Data yang Digunakan berasal dari dataset SKLearn yaitu California Housing Dataset. Kode berikut berfungsi untuk mengambil dataset menggunakan fungsi fetch_california_housing dari library sklearn.datasets
"""

data = fetch_california_housing(as_frame=True)
df = data.frame

print(df.shape)  # Output: (20640, 9)
df

"""Dataset ini berisi 20.640 baris data dan 9 kolom (8 fitur dan 1 target):
- MedInc: Median income di blok tersebut
- HouseAge: Umur median rumah
- AveRooms: Rata-rata jumlah kamar per rumah tangga
- AveBedrms: Rata-rata jumlah kamar tidur
- Population: Jumlah populasi di area
- AveOccup: Rata-rata jumlah penghuni per rumah
- Latitude: Koordinat geografis (lintang)
- Longitude: Koordinat geografis (bujur)
- MedHouseVal: Median nilai rumah di blok tersebut (dalam satuan $100.000) -> Fitur Target

Dikarenakan variabel Latitude dan Longitude merupakan variabel kesatuan, maka kita akan mengubahnya dulu menjadi variabel DistanceToLA, yaitu jarak rumah tersebut ke pusat kota (Los Angeles) agar model ML yang digunakan nantinya dapat menginterpretasikan variabel ini lebih mudah.
"""

city_lat = 34.05     # Los Angeles latitude
city_lon = -118.25   # Los Angeles longitude

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius bumi dalam kilometer
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

data = fetch_california_housing(as_frame=True)
df = data.frame

# Tambahkan kolom jarak ke pusat kota
df['DistanceToLA'] = haversine(df['Latitude'], df['Longitude'], 34.05, -118.25)
df = df.drop(columns=['Latitude', 'Longitude'])
print("Tampilan data setelah dilakukan perubahan variabel:")
df

"""Formula Haversine adalah rumus matematika yang digunakan untuk menghitung jarak terpendek (great-circle distance) antara dua titik di permukaan bumi berdasarkan lintang (latitude) dan bujur (longitude), dengan asumsi bahwa bumi berbentuk bulat sempurna. Parameter 34.05 dan 118.25 adalah latitude dan longitude untuk kota Los Angeles secara berurutan.

**EDA - Missing Value**

Untuk memeriksa apakah terdapat missing value, jalankan kode berikut:
"""

df.info()

"""Berdasarkan output yang tampil, tidak terlihat adanya missing value, namun kita akan memeriksa lebih mendalam menggunakan kode df.describe()."""

df.describe()

"""Pada baris min di kolom DistanceToLA, terdapat nilai 0. Untuk memeriksa seberapa banyak data yang memiliki nilai 0 ini, kita akan menjalankan kode berikut:"""

DistanceToLA = (df.DistanceToLA == 0).sum()
print("Nilai 0 di kolom DistanceToLA ada: ", DistanceToLA)

"""Diketahui bahwa nilai 0 di kolom DistanceToLA hanya ada 2, jumlah ini tergolong kecil dibandingkan jumlah keseluruhan baris, maka dari itu baris tersebut akan dihapus pada tahap Data Preparation.

**EDA - Outlier**

Untuk memeriksa nilai outlier, kita perlu memvisualisasikan persebaran data setiap kolom menggunakan boxplot dengan kode berikut pada setiap kolom:
"""

sns.boxplot(x=df['MedInc'])

sns.boxplot(x=df['HouseAge'])

sns.boxplot(x=df['AveRooms'])

sns.boxplot(x=df['AveBedrms'])

sns.boxplot(x=df['Population'])

sns.boxplot(x=df['AveOccup'])

sns.boxplot(x=df['MedHouseVal'])

sns.boxplot(x=df['DistanceToLA'])

"""Berdasarkan gambar tersebut dapat diketahui bahwa terdapat outlier pada variabel MedInc, AveRooms, AveBedrms, Population, AveOccup, dan MedHouseVal. Maka dari itu, pada proses Data Preparation kita akan menangani masalah ini menggunakan teknik winsorizing.

**EDA - Univariate Analysis**

Untuk melihat visualisasi univariate analysis, kita dapat melakukannya dengan menerapkan visualisasi histogram dengan kode berikut:
"""

numerical_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'MedHouseVal', 'DistanceToLA']

df.hist(bins=50, figsize=(20,15))
plt.show()

"""Pada variabel target, yaitu variabel MedHouseVal, dapat dilihat bahwa:

- Peningkatan harga rumah sebanding dengan penurunan jumlah sampel. Hal ini dapat kita lihat jelas dari histogram "MedHouseVal" yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu y). Namun terdapat suatu harga di antara 400.0000 - 500.000 Dollar AS yang memiliki sampel yang tinggi.
- Rentang harga rumah cukup beragam yaitu dari skala puluhan ribu dolar hingga >$500.000 AS.
- Distribusi harga miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model.

**EDA - Multivariate Analysis**

Untuk mengamati hubungan antara fitur numerik, kita akan menggunakan fungsi pairplot() dengan kode berikut:
"""

sns.pairplot(df, diag_kind = 'kde')

"""Variabel MedHouseVal yang menjadi variabel target berada di baris ke-7. Sebaran data yang terlihat pada plot masih acak, kecuali pada variabel MedInc dan DistanceToLA. Variabel MedInc terlihat berkorelasi positif dengan variabel target, sedangkan variabel DistanceToLA terlihat berkorelasi negatif terbalik dengan variabel target. Artinya, semakin tinggi median pendapatan maka semakin tinggi juga harga perumahan, dan semakin jauh perumahan tersebut dari pusat kota, maka harga perumahan semakin rendah. Untuk memperjelas nilai korelasi seluruh variabel numerik dengan variabel target, kita akan menggunakan visualisasi Correlation Matrix dengan kode berikut:"""

plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_features].corr().round(2)

# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

"""Setelah diamati, variabel yang memiliki nilai korelasi tertinggi adalah variabel MedInc (korelasi positif), AveRooms (korelasi positif), AveOccup (korelasi negatif), DistanceToLA (korelasi negatif), HouseAge (korelasi positif), dan AveBedrms (korelasi negatif) secara berurutan.

Data Preparation
---

Pada tahap ini akan dilakukan penanganan missing value dan outlier dan pemilihan variabel yang paling berkolerasi terhadap variabel target, yaitu MedHouseVal. Selanjutnya, kita akan melakukan tahapan Train-Test Split dan Standardisasi

**Penanganan Missing Value**

Berdasarkan informasi yang didapatkan pada tahap Data Understanding, jumlah baris yang memiliki missing value hanya ada 2. Maka dari itu kita akan menghapus baris tersebut karena hanya sebagian kecil dari jumlah data yang ada.
"""

# Drop baris dengan nilai 'DistanceToLA' = 0
df = df.loc[(df[['DistanceToLA']]!=0).all(axis=1)]

# Cek ukuran data untuk memastikan baris sudah di-drop
df.shape

"""Dengan begitu, maka jumlah data yang kita miliki sekarang adalah 20,638 data.

**Penanganan Outlier**

Untuk menangani nilai outlier kita akan melakukan teknik winsorizing, yaitu mengubah nilai outlier menjadi nilai ambang atas atau ambang bawah, sehingga tidak mengurangi data yang sudah ada. Teknik winsorizing dapat dilakukan dengan menerapkan kode berikut:
"""

# Ambil hanya kolom numerikal
numeric_cols = df.select_dtypes(include='number').columns

# Hitung Q1, Q3, dan IQR untuk kolom numerikal
Q1 = df[numeric_cols].quantile(0.25)
Q3 = df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1

# Hitung batas bawah dan atas
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Terapkan winsorizing: batasi nilai-nilai ekstrem
df[numeric_cols] = df[numeric_cols].clip(lower=lower_bound, upper=upper_bound, axis=1)

# Cek ukuran dataset (tidak berubah karena tidak menghapus baris)
df.shape

"""Di mana `numeric_cols` adalah variabel yang bertipe data numerik, `lower_bound` adalah batas bawah, dan `upper_bound` adalah batas atas. Jika ada nilai yang melewati batas bawah, maka nilai tersebut akan dijadikan nilai batas bawah, begitu juga dengan kasus outlier yang melewati batas atas.

**Pemilihan Variabel**

Berdasarkan visualisasi Correlation Matrix seluruh variabel numerik terhadap variabel target (MedHouseVal), variabel Population memiliki nilai korelasi yang rendah, yaitu -0,03 (kurang dari ±0.1), sehingga variabel ini akan dihapus dan tidak diikutsertakan dalam perhitungan. Hal ini dapat dilakukan menggunakan kode berikut:
"""

# Drop kolom Population karena memiliki tingkat korelasi yang sangat rendah
df.drop(['Population'], inplace=True, axis=1)
df.head()

"""**Train-Test Split**

Pembagian data training-data testing menggunakan rasio 8:2 karena jumlah data yang tersedia tergolong cukup, sehingga total data training akan berjumlah 16510 dan data testing berjumlah 4128 dari keseluruhan 20638 total data (setelah dilakukan penghapusan pada baris yang memiliki missing value). Pembagian ini dilakukan dengan kode berikut:
"""

from sklearn.model_selection import train_test_split

X = df.drop(["MedHouseVal"],axis =1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""Variabel X menampung kolom-kolom yang digunakan sebagai nilai yang menentukan kolom target, yaitu MedHouseVal. Variabel y menampung kolom target, yaitu MedHouseVal.

**Standardisasi**

Standardisasi diterapkan agar data memiliki skala relatif sama atau mendekati disrtibusi normal. Hal ini membuat model machine learning memiliki performa dan konvergensi yang lebih baik. Standardisasi hanya diterapkan pada data training untuk menghindari kebocoran informasi pada data testing dengan menggunakan teknik StandardScaler. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1. Teknik ini dapat diterapkan dengan kode berikut:
"""

from sklearn.preprocessing import StandardScaler

numerical_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'AveOccup', 'DistanceToLA']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])

# Tampilkan 5 index data training teratas yang sudah diterapkan standardisasi
X_train[numerical_features].head()

"""Setelah standardisasi diterapkan, maka tabel statistik deskriptif dari data training menjadi seperti ini:"""

X_train[numerical_features].describe().round(4)

"""Modelling
---

Untuk menyelesaikan permasalahan, tiga model machine learning akan digunakan, yaitu K-Nearest Neighbor (KNN), Random Forest (RF), dan Adaptive Boosting (AdaBoost). Pertama, kita akan menyiapkan dataframe terlebih dahulu untuk analisis model dengan kode berikut:
"""

# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'Boosting'])

"""**K-Nearest Neighbor (KNN)**

K-Nearest Neighbor adalah algoritma berbasis instance-based learning, yang berarti ia tidak membentuk model eksplisit melainkan membuat prediksi berdasarkan kedekatan terhadap data latih. Dalam konteks regresi, KNN akan mencari sejumlah k tetangga terdekat dari data uji berdasarkan metrik jarak (umumnya Euclidean), kemudian memprediksi nilai target dengan rata-rata dari target tetangga-tetangga tersebut. Semakin kecil nilai k, model menjadi lebih kompleks dan sensitif terhadap noise (overfitting), sementara nilai k yang besar dapat menyebabkan underfitting.

Pada implementasi ini, digunakan nilai n_neighbors=10 dan pengukuran jarak Euclidean distance (bawaan dari library SKLearn), yang berarti prediksi didasarkan pada 10 tetangga terdekat. Pemilihan nilai k yang terlalu kecil dapat menyebabkan model overfitting, karena terlalu sensitif terhadap noise, sedangkan nilai k yang terlalu besar dapat menyebabkan underfitting karena prediksi menjadi terlalu umum.

Untuk melatih model KNN, dapat dilakukan dengan kode berikut:
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

"""**Random Forest (RF)**

Random Forest merupakan algoritma ensemble learning berbasis bagging, yang menggabungkan banyak decision tree untuk meningkatkan akurasi dan mengurangi overfitting. Setiap pohon dibangun dari subset acak data latih dan subset fitur, sehingga menghasilkan pohon yang saling independen. Pada regresi, hasil prediksi adalah rata-rata dari semua prediksi pohon-pohon tersebut. Kelebihan Random Forest adalah kemampuannya menangani data dengan banyak fitur dan menghindari overfitting lebih baik dibanding single decision tree.

Pada model ini, digunakan beberapa parameter penting, yaitu:
- n_estimators=50: jumlah pohon dalam forest adalah 50.
- max_depth=16: kedalaman maksimum tiap pohon adalah 16, yang digunakan untuk mengontrol kompleksitas model.
- random_state=55: untuk memastikan hasil yang dapat direproduksi.
- n_jobs=-1: memanfaatkan seluruh inti CPU untuk mempercepat pelatihan.
Dengan konfigurasi ini, Random Forest dapat menangani data berdimensi tinggi dan memiliki kemampuan generalisasi yang lebih baik dibanding decision tree tunggal.

Untuk melatih model RF, dapat dilakukan dengan kode berikut:
"""

from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

"""**Adaptive Boosting (AdaBoost)**

AdaBoost atau Adaptive Boosting adalah metode ensemble learning yang menggabungkan beberapa model lemah (weak learners), biasanya decision stumps (pohon dengan satu split), secara berurutan. Setiap model selanjutnya berfokus pada kesalahan model sebelumnya dengan memberikan bobot lebih besar pada data yang sulit diprediksi. Dalam regresi, model lemah mencoba meminimalkan kesalahan dengan cara menyesuaikan prediksi terhadap residual (selisih antara nilai sebenarnya dan prediksi sebelumnya). Kelebihan AdaBoost adalah kemampuannya untuk meningkatkan akurasi model meskipun hanya menggunakan model lemah, namun ia sensitif terhadap outlier.

Dalam implementasi ini, digunakan parameter:
- learning_rate=0.05, yang mengontrol kontribusi masing-masing model terhadap prediksi akhir. Nilai yang lebih kecil membuat model belajar lebih perlahan dan dapat mencegah overfitting.
- random_state=55, untuk hasil yang konsisten saat proses pelatihan diulang.
Meskipun hanya menggunakan model lemah, AdaBoost mampu meningkatkan akurasi secara signifikan. Namun, algoritma ini cenderung sensitif terhadap data outlier karena bobot yang diberikan bisa menjadi sangat besar untuk data yang sulit diprediksi.

Untuk melatih model AdaBoost, dapat dilakukan dengan kode berikut:
"""

from sklearn.ensemble import AdaBoostRegressor

boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

"""Evaluasi
---

Ketiga model yang telah disebutkan akan dievaluasi menggunakan metrik Mean Squared Error (MSE) untuk menentukan performa terbaik. MSE adalah ukuran rata-rata dari kuadrat selisih antara nilai prediksi model dengan nilai sebenarnya. Semakin kecil nilai MSE, semakin baik performa model dalam memprediksi data, karena menunjukkan bahwa prediksi model mendekati nilai aktual.

MSE mengukur seberapa jauh prediksi model dari nilai sebenarnya. Nilai MSE yang lebih rendah menandakan bahwa model memiliki kesalahan prediksi yang lebih kecil.

Sebelum melakukan evaluasi, data testing akan diperlakukan sama seperti data training, yaitu menerapkan standardisasi dari StandardScaler agar datanya memiliki nilai rata-rata = 0 dan varians = 1. Hal ini dilakukan agar model dapat mengenali data testing karena sebelumnya model dilatih menggunakan nilai yang telah distandardisasi. Kode untuk menerapkannya sebagai berikut:
"""

# Lakukan scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

"""Untuk menjelaskan hasil evaluasi model pada saat training dan testing, kita akan membuat sebuah variabel dengan nama mse. Selanjutnya kita akan mengevaluasi ketiga model pada data testing menggunakan kode lengkapnya seperti berikut:"""

# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3

# Panggil mse
mse

"""Dan jika divisualisasikan dengan horizontal barplot akan seperti ini:"""

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

"""Secara keseluruhan, ketiga model memberikan nilai MSE pada proses training dan testing relatif kecil. Random Forest (RF) memberikan nilai eror yang paling kecil, namun perbandingan nilai MSE antara proses training dan testing cukup jauh, ini menunjukkan bahwa model tersebut mengalami masalah overfitting. K-Nearest Neighbor (KNN) memberikan nilai eror terkecil kedua, namun masih terdapat perbedaan yang cukup signifikan pada nilai MSE antara proses training dan testing. Sedangkan model AdaBoost memiliki eror yang paling besar (berdasarkan grafik, angkanya di atas 0.0005), namun nilai MSE yang diperoleh oleh model ini pada proses training dan testing tidak berbeda jauh, sehingga dapat dikatakan bahwa model inilah yang paling stabil.

Untuk mengujinya, kita akan membuat prediksi menggunakan kode berikut:
"""

prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)

"""Data yang diprediksi oleh kode tersebut adalah data dengan index ke-4414 dengan nilai y_true (MedHouseVal atau harga asli) bernilai 1.743. Nilai yang diprediksi oleh KNN yaitu 1.9, RF 2.3, dan AdaBoost 1.8. Dapat dilihat bahwa AdaBoost memprediksi nilai yang terdekat dengan nilai aslinya.

Untuk menjawab Problem Statements dan Goals pada proses Business Understanding, kita akan mengevaluasi masalah tersebut secara bertahap.

### Problem Statements - Jawaban
1. Bagaimana hubungan antara faktor demografis (seperti kepadatan penduduk dan tingkat pendapatan) dengan harga rumah?
- Jawaban: Berdasarkan variabel pada dataset California Housing, variabel yang merepresentasikan faktor demografis seperti pendapatan (MedInc) dan populasi (Population dan AveOccup) memiliki pengaruh yang cukup signifikan dalam menentukan harga rumah. Untuk faktor pendapatan (MedInc), variabel ini berkorelasi positif terhadap harga rumah, dengan kata lain, semakin tinggi pendapatan penduduk di perumahan tersebut, maka harga rumah di wilayah tersebut juga semakin tinggi. Untuk faktor populasi (Popluation dan AveOccup), variabel tersebut berkorelasi negatif terhadap harga rumah, dengan kata lain, semakin tinggi populasi yang ada di perumahan tersebut, maka harga rumah di wilayah tersebut juga semakin rendah.
2. Dapatkah kita memprediksi harga median rumah di suatu daerah menggunakan data demografis dan geografis?
- Jawaban: Ya, kita dapat memprediksi harga median rumah di suatu daerah menggunakan data demografis dan geografis. Beberapa variabel yang digunakan dalam dataset pada penelitian ini sudah merepresentasikan faktor demografis dan geografis. Ada pun variabel yang merepresentasikan faktor demografis adalah pendapatan (MedInc) dan populasi (Population dan AveOccup). Sedangkan faktor geografis direpresentasikan oleh variabel Latitude dan Longitude (yang kemudian diubah menjadi variabel yang merepresentasikan jarak ke pusat kota, yaitu variabel DistanceToLA). Namun, lebih baik jika kita menambahkan variabel yang merepresentasikan faktor properti seperti jumlah kamar dan usia rumah agar hasil prediksi menjadi lebih baik.
3. Fitur apa yang paling berpengaruh terhadap harga rumah di California?
- Jawaban: Berdasarkan tahapan pada Data Understanding, khususnya visualisasi variabel numerik terhadap variabel target, fitur yang paling berpengaruh adalah fitur MedInc (korelasi positif), AveRooms (korelasi positif), AveOccup (korelasi negatif), DistanceToLA (korelasi negatif), HouseAge (korelasi positif), dan AveBedrms (korelasi negatif) secara berurutan.

### Goals - Jawaban
1. Mengetahui fitur-fitur yang secara signifikan berkorelasi dengan harga rumah.
- Jawaban: Ya, tujuan ini sudah tercapai pada saat proses memvisualisasikan pengaruh variabel numerik terhadap variabel target dengan visualisasi Correlation Matrix.
2. Membangun model prediksi harga rumah berdasarkan fitur yang tersedia.
- Jawaban: Ya tujuan ini sudah tercapai pada tahap Evaluasi, khususnya bagian Prediksi yang menggunakan 3 model (KNN, RF, dan AdaBoost) untuk memprediksi harga rumah dengan jumlah data tertentu.
3. Menganalisis pentingnya masing-masing fitur untuk memahami kontribusinya terhadap nilai harga rumah.
- Jawaban: Ya, tujuan ini sudah tercapai pada saat proses memvisualisasikan pengaruh variabel numerik terhadap variabel target dengan visualisasi Correlation Matrix.

Setiap solusi statement yang digunakan pada penelitian ini berdampak, mulai dari perubahan format data (Latitude dan Longitude menjadi DistanceToLA) agar model memahami variabel lebih baik, penanganan data yang hilang, outlier, dan pemilihan variabel yang signifikan. Hasil ini ditunjukkan oleh nilai Mean Squared Error (MSE) yang rendah oleh ketiga model, yaitu di bawah
0.0006.
"""