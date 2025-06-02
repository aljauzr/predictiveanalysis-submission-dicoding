# Laporan Proyek Machine Learning - Al Jauzi Abdurrohman

## Domain Proyek

Permasalahan dalam sektor properti, khususnya perumahan, merupakan topik penting baik bagi individu maupun instansi. Di negara bagian California, harga rumah mengalami fluktuasi yang dipengaruhi oleh berbagai faktor seperti jumlah penduduk, tingkat pendapatan, dan lokasi geografis. Oleh karena itu, penting untuk memahami faktor-faktor apa saja yang dapat memengaruhi harga rumah agar dapat membantu calon pembeli, pengembang properti, hingga investor dalam membuat keputusan yang lebih tepat.

Dalam proyek ini, akan dilakukan analisis dan pembuatan model prediksi harga rumah berdasarkan data demografis dan geografis dari California Housing Dataset. Dataset ini merepresentasikan kondisi blok-blok sensus pada tahun 1990 di California dan sering digunakan untuk kasus regresi di bidang data science dan real estate.

## Business Understanding

### Problem Statements
1. Bagaimana hubungan antara faktor demografis (seperti kepadatan penduduk dan tingkat pendidikan) dengan harga rumah?
2. Dapatkah kita memprediksi harga median rumah di suatu daerah menggunakan data demografis dan geografis?
3. Fitur apa yang paling berpengaruh terhadap harga rumah di California?

### Goals
1. Mengetahui fitur-fitur yang secara signifikan berkorelasi dengan harga rumah.
2. Membangun model prediksi harga rumah berdasarkan fitur yang tersedia.
3. Menganalisis pentingnya masing-masing fitur untuk memahami kontribusinya terhadap nilai harga rumah.

## Data Understanding
Dataset yang digunakan adalah California Housing Dataset dari sklearn.datasets, yang juga tersedia di Kaggle California Housing Dataset dan berasal dari California Census tahun 1990 (URL: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). Pada sel kode notebook, dataset ini dapat diunduh menggunakan kode berikut:
```sh
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame
```
Dataset ini berisi 20.640 baris data dan 9 kolom (8 fitur dan 1 target).

### Variabel-variabel pada California Housing Dataset adalah sebagai berikut:

| Kolom | Tipe Data | Deskripsi |
| ------ | ------ | ------ |
| MedInc | float64 | Median pendapatan dalam blok (dalam puluhan ribu dolar AS). |
| HouseAge | float64 | Umur median rumah di blok. |
| AveRooms | float64 | Rata-rata jumlah kamar per rumah. |
| AveBedrms | float64 | Rata-rata jumlah kamar tidur per rumah. |
| Population | float64 | Jumlah penduduk dalam blok. |
| AveOccup | float64 | Rata-rata jumlah penghuni per rumah. |
| Latitude | float64 | Koordinat geografis lintang. |
| Longitude | float64 | Koordinat geografis bujur. |
| MedHouseVal | float64 | Median nilai rumah dalam satu blok (dalam ratusan ribu dolar AS). |

Variabel Latitude dan Longitude merupakan variabel kesatuan, agar model ML yang digunakan dapat menginterpretasikan variabel ini lebih mudah, maka pada tahap Data Preparation kita akan mengubah kedua variabel tersebut menjadi satu variabel: DistanceToLA, yaitu jarak ke pusat kota (Los Angeles).

## Data Preparation
Pada tahap ini akan dilakukan proses Exploratory Data Analysis yang mencakup pemeriksaan missing value, nilai outlier, univariate analysis, dan multivariate analysis. Setelah tahap EDA dilakukan, selanjutnya kita akan melakukan tahapan Train-Test Split dan Standardisasi untuk mempersiapkan data agar dapat diolah oleh model machine learning.

Namun, sebelum masuk ke tahap EDA, kita akan mengubah variabel Latitude dan Longitude menjadi satu variabel DistanceToLA yang merepresentasikan jarak ke pusat kota (Los Angeles) menggunakan formula haversine dengan kode berikut:
```sh
df['DistanceToLA'] = haversine(df['Latitude'], df['Longitude'], 34.05, -118.25)
```
Formula Haversine adalah rumus matematika yang digunakan untuk menghitung jarak terpendek (great-circle distance) antara dua titik di permukaan bumi berdasarkan lintang (latitude) dan bujur (longitude), dengan asumsi bahwa bumi berbentuk bulat sempurna. Parameter `34.05` dan `118.25` adalah latitude dan longitude untuk kota Los Angeles secara berurutan.
### EDA - Missing Value
Untuk memeriksa apakah terdapat missing value, jalankan kode berikut:
```sh
df.describe()
```
Output yang tampil seperti ini:
| | Medinc | HouseAge | AveRooms | AveBedrms | Population | AveOccup | MedHouseVal | DistanceToLA
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| **count** | 20640.000000 | 20640.000000 | 20640.000000 | 20640.000000 | 20640.000000 | 20640.000000 | 20640.000000 | 20640.000000 |
| **mean** | 3.870671 | 28.639486 | 5.429000 | 1.096675 | 1425.476744 | 3.070655	| 2.068558 | 269.411741 |
| **std** | 1.899822 | 12.585558 | 2.474173	| 0.473911 | 1132.462122 | 10.386050 | 1.153956 | 247.652206 | 
| **min** | 0.499900 | 1.000000 | 0.846154 | 0.333333 | 3.000000 | 0.692308	| 0.149990 | 0.000000 |
| **25%** | 2.563400 | 18.000000 | 4.440716 | 1.006079 | 787.000000 | 2.429741 | 1.196000 | 32.223307|
| **50%** | 3.534800 | 29.000000 | 5.229129 | 1.048780 | 1166.000000 | 2.818116 | 1.797000 | 173.825911 |
| **75%** | 4.743250 | 37.000000 | 6.052381 | 1.099526 | 1725.000000 | 3.282261 | 2.647250 | 526.985985 |
| **max** | 15.000100 | 52.000000 | 141.909091 | 34.066667 | 35682.000000 | 1243.333333 | 5.000010 | 1018.198911 |

Terdapat missing value (bernilai 0) pada kolom DistanceToLA dan setelah dilakukan pemeriksaan, baris dengan missing value tersebut hanya ada dua. Jumlah ini tergolong kecil dibandingkan jumlah keseluruhan baris, maka dari itu baris tersebut akan dihapus.

### EDA - Outlier
Untuk memeriksa nilai outlier, kita perlu memvisualisasikan persebaran data setiap kolom menggunakan boxplot dengan kode berikut:
```sh
sns.boxplot(x=df['{namaKolom}'])
```
Hasil untuk seluruh kolom ditampilkan pada gambar berikut:
![Boxplots](images/Boxplots.png)

Berdasarkan gambar tersebut dapat diketahui bahwa terdapat outlier pada variabel MedInc, AveRooms, AveBedrms, Population, AveOccup, dan MedHouseVal. Maka dari itu, kita akan melakukan teknik winsorizing, yaitu mengubah nilai outlier menjadi nilai ambang atas atau ambang bawah, sehingga tidak mengurangi data yang sudah ada.
Teknik winsorizing dapat dilakukan dengan menerapkan kode berikut:
```sh
df[numeric_cols] = df[numeric_cols].clip(lower=lower_bound, upper=upper_bound, axis=1)
```
Di mana `numeric_cols` adalah variabel yang bertipe data angka, `lower_bound` adalah batas bawah, dan `upper_bound` adalah batas atas. Jika ada nilai yang melewati batas bawah, maka nilai tersebut akan dijadikan nilai batas bawah, begitu juga dengan kasus outlier yang melewati batas atas.

### EDA - Univariate Analysis
Untuk melihat visualisasi univariate analysis, kita dapat melakukannya dengan menerapkan visualisasi histogram dengan kode berikut:
```sh
df.hist(bins=50, figsize=(20,15))
plt.show()
```
Sehingga menghasilkan gambar berikut:
![EDA - Univariate Analysis](images/EDA%20-%20Univariate%20Analysis.png)

Pada variabel target, yaitu variabel MedHouseVal, dapat dilihat bahwa:
- Peningkatan harga rumah sebanding dengan penurunan jumlah sampel. Hal ini dapat kita lihat jelas dari histogram "MedHouseVal" yang grafiknya mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu y). Namun terdapat suatu harga di antara $400.0000 - $500.000 AS yang memiliki sampel yang tinggi.
- Rentang harga rumah cukup beragam yaitu dari skala puluhan ribu dolar hingga >$500.000 AS.
- Distribusi harga miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model.
### EDA - Multivariate Analysis
Untuk mengamati hubungan antara fitur numerik, kita akan menggunakan fungsi pairplot() dengan kode berikut:
```sh
sns.pairplot(df, diag_kind = 'kde')
```
Sehingga menghasilkan gambar berikut:
![EDA - Multivariate Analysis](images/EDA%20-%20Multivariate%20Analysis.png)

Variabel MedHouseVal yang menjadi variabel target berada di baris ke-7. Sebaran data yang terlihat pada plot masih acak, kecuali pada variabel MedInc dan DistanceToLA. Variabel MedInc terlihat berkorelasi positif dengan variabel target, sedangkan variabel DistanceToLA terlihat berkorelasi negatif terbalik dengan variabel target. Artinya, semakin tinggi median pendapatan maka semakin tinggi juga harga perumahan, dan semakin jauh perumahan tersebut dari pusat kota, maka harga perumahan semakin rendah.

Untuk memperjelas nilai korelasi seluruh variabel numerik dengan variabel target, kita akan menggunakan visualisasi correlation matrix dengan kode berikut:
```sh
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_features].corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)
```
Sehingga menghasilkan gambar berikut:
![Correlation Matrix](images/Correlation%20Matrix.png)

Setelah diamati, variabel yang memiliki nilai korelasi tertinggi adalah variabel MedInc (korelasi positif), AveRooms (korelasi positif), AveOccup (korelasi negatif), DistanceToLA (korelasi negatif), HouseAge (korelasi positif), dan AveBedrms (korelasi negatif) secara berurutan. Variabel Population memiliki nilai korelasi yang rendah, yaitu -0,03 (kurang dari ±0.1), sehingga variabel ini akan dihapus dan tidak diikutsertakan dalam perhitungan dengan kode berikut:
```sh
df.drop(['Population'], inplace=True, axis=1)
```
### Train-Test Split
Pembagian data training-data testing menggunakan rasio 8:2 karena jumlah data yang tersedia tergolong cukup, sehingga total data training akan berjumlah 16510 dan data testing berjumlah 4128 dari keseluruhan 20638 total data (setelah dilakukan penghapusan pada baris yang memiliki missing value). Pembagian ini dilakukan dengan kode berikut:
```sh
from sklearn.model_selection import train_test_split

X = df.drop(["MedHouseVal"],axis =1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)
```
Variabel X menampung kolom-kolom yang digunakan sebagai nilai yang menentukan kolom target, yaitu MedHouseVal. Variabel y menampung kolom target, yaitu MedHouseVal.
### Standardisasi
Standardisasi diterapkan agar data memiliki skala relatif sama atau mendekati disrtibusi normal. Hal ini membuat model machine learning memiliki performa dan konvergensi yang lebih baik. Standardisasi hanya diterapkan pada data training untuk menghindari kebocoran informasi pada data testing dengan menggunakan teknik StandardScaler. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1. Teknik ini dapat diterapkan dengan kode berikut:
```sh
from sklearn.preprocessing import StandardScaler

numerical_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'AveOccup', 'DistanceToLA']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
```
Setelah standardisasi diterapkan, maka table statistik deskriptif dari data training menjadi seperti ini:
| | Medinc | HouseAge | AveRooms | AveBedrms | AveOccup | DistanceToLA |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| **count** | 16510.0000 | 16510.0000 | 16510.0000 | 16510.0000 | 16510.0000 | 16510.0000 |
| **mean** | -0.0000 | 0.0000 | -0.0000 | 0.0000 | 0.0000 |	0.0000 |
| **std** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **min** | -1.9864 | -2.2001 | -2.6294 | -2.3873 | -2.5302 | -1.0807|
| **25%** | -0.7443 | -0.8479 | -0.6950 | -0.6506 | -0.6792 | -0.9549 |
| **50%** | -0.1635 | 0.0270 | -0.0587 | -0.1158 | -0.1154 | -0.3863 |
| **75%** | 0.5637 | 0.6634 | 0.5989 | 0.5201 | 0.5515 | 1.0405 |
| **max** | 2.5382 | 1.8565 | 2.5422 | 2.2489 | 2.4099 | 3.0317 |

## Modelling
Untuk menyelesaikan permasalahan, tiga model machine learning akan digunakan, yaitu K-Nearest Neighbor (KNN), Random Forest (RF), dan Adaptive Boosting (AdaBoost).
Pertama, kita akan menyiapkan dataframe terlebih dahulu untuk analisis model dengan kode berikut:
```sh
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'Boosting'])
```                      
### K-Nearest Neighbor (KNN)
Untuk melatih model KNN, dapat dilakukan dengan kode berikut:
```sh
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)
```
### Random Forest (RF)
Untuk melatih model RF, dapat dilakukan dengan kode berikut:
```sh
from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)
```
### Adaptive Boosting (AdaBoost)
Untuk melatih model AdaBoost, dapat dilakukan dengan kode berikut:
```sh
from sklearn.ensemble import AdaBoostRegressor

boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)
```

## Evaluation
Ketiga model yang telah disebutkan akan dievaluasi menggunakan metrik Mean Squared Error (MSE) untuk menentukan performa terbaik. MSE adalah ukuran rata-rata dari kuadrat selisih antara nilai prediksi model dengan nilai sebenarnya. Semakin kecil nilai MSE, semakin baik performa model dalam memprediksi data, karena menunjukkan bahwa prediksi model mendekati nilai aktual. MSE dirumuskan sebagai berikut:
![Formula MSE](images/Formula.png)

Di mana:
- n = jumlah data (observasi)
- yi = nilai aktual (true value) ke-i
- ŷi = nilai prediksi model ke-i
- (yi-ŷi)^2 = selisih kuadrat antara nilai aktual dan prediksi

MSE mengukur seberapa jauh prediksi model dari nilai sebenarnya. Nilai MSE yang lebih rendah menandakan bahwa model memiliki kesalahan prediksi yang lebih kecil.

Sebelum melakukan evaluasi, data testing akan diperlakukan sama seperti data training, yaitu menerapkan standardisasi dari StandardScaler agar datanya memiliki nilai rata-rata = 0 dan varians = 1. Hal ini dilakukan agar model dapat mengenali data testing karena sebelumnya model dilatih menggunakan nilai yang telah distandardisasi. Kode untuk menerapkannya sebagai berikut:
```sh
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])
```
Untuk menjelaskan hasil evaluasi model pada saat training dan testing, kita akan membuat sebuah variabel dengan nama mse dengan kode berikut:
```sh
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])
```
Selanjutnya kita akan mengevaluasi ketiga model pada data testing menggunakan kode berikut:
```sh
# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3
```
Hasil dari evaluasi ketiga model tersebut adalah sebagai berikut:
| | train | test |
| ------ | ------ | ------ |
| **KNN** | 0.000328 | 0.0003760 |
| **RF** | 0.000076 | 0.000352 |
| **AdaBoost** | 0.000536 | 0.000531 |

Dan jika divisualisasikan dengan horizontal barplot akan seperti ini:
![MSE](images/MSE.png)

Secara keseluruhan, ketiga model memberikan nilai MSE pada proses training dan testing relatif kecil. Random Forest (RF) memberikan nilai eror yang paling kecil, namun perbandingan nilai MSE antara proses training dan testing cukup jauh, ini menunjukkan bahwa model tersebut mengalami masalah overfitting. K-Nearest Neighbor (KNN) memberikan nilai eror terkecil kedua, namun masih terdapat perbedaan yang cukup signifikan pada nilai MSE antara proses training dan testing. Sedangkan model AdaBoost memiliki eror yang paling besar (berdasarkan grafik, angkanya di atas 0.0005), namun nilai MSE yang diperoleh oleh model ini pada proses training dan testing tidak berbeda jauh, sehingga dapat dikatakan bahwa model inilah yang paling stabil.

Untuk mengujinya, kita akan membuat prediksi menggunakan kode berikut:
```sh
prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)
```
Hasil yang ditampilkan seperti berikut:
| | y_true | prediksi_KNN | prediksi_RF | prediksi_Boosting |
| ------ | ------ | ------ | ------ | ------ |
| **4414** | 1.743 | 1.9 | 2.3 | 1.8 |

Data yang diprediksi oleh kode tersebut adalah data dengan index ke-4414 dengan nilai y_true (MedHouseVal atau harga asli) bernilai 1.743. Nilai yang diprediksi oleh KNN yaitu 1.9, RF 2.3, dan AdaBoost 1.8. Dapat dilihat bahwa AdaBoost memprediksi nilai yang terdekat dengan nilai aslinya.

**---Ini adalah bagian akhir laporan---**
