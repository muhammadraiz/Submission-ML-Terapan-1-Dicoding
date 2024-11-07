# Laporan Proyek Machine Learning - Raiz
## Domain Proyek
Perkembangan industri membuat penggunaan mesin dalam sektor produksi menjadi meningkat. Tuntutan hasil yang minim cacat serta efisiensi waktu membuat mesin menjadi solusi paling primadona dalam tahap produksi. Sejauh ini ada sebagian besar proses produksi sudah memanfaatkan mesin untuk mengerjakannya. Namun masi ada beberapa sektor yang masi membutuhkan keberadaan manusia, bahkan proses operasi mesin juga membutuhkan manusia sebagai operator.

Penggunaan mesin memang memberikan banyak kelebihan baik dari segi kualitas, konsistensi, maupun waktu. Meskipun begitu, seperti halnya sumber daya manusia, mesin tidak selamanya sempurna melihat kapasitasnya pasti memiliki sebauh limitasi. Dalam kurun waktu yang cukup lama, kinerja mesin cenderung akan menurun, hal ini disebabkan oleh banyak faktor dibelakangnya baik itu disengaja maupun tidak. Oleh karena itu adakalanya sebuah mesin akan mengalami proses perbaikan atau disebut maintenance.

Permasalahannya adalah tidak mungkin setiap waktu kita memonitor kinerja mesin tersebut dan membandingkannya dengan data yang sudah ada setiap harinya. Proses comparison ini juga terkadang membutuhkan perhitungan ahli untuk melakukannya. Oleh karena itu, dibutuhkan suatu sistem pintar yang mampu menghitung dan menilai kinerja suatu mesin berdasarkan beberapa variabel.

## Bussiness Understanding
### Problem Statement
Berdasarkan uraian terkait kondisi yang dialami, ada beberapa permasalahan yang dapat dijabarkan untuk dicari solusinya. Permasalahan tersebut dijabarkan sebagai berikut:
- Variabel apa yang menjadi titik paling krusial dalam _failure_ nya suatu mesin?
- Kapan sebuah mesin memerluhkan proses maintenance dan pada kondisi seperti apa?
### Goals
- Mencari variabel yang paling berkorelasi terhadap kondisi _failure_
- Membuat sebuah sistem klasifikasi kondisi mesin dari beberapa variabel yang ada pada mesin tersebut
### Solution Statement
- Menggunakan correlation matrix dari scikit-learn untuk melihat variabel yang nilai korelasinya paling jauh dari 0
- Menggunakan baseline model sebagai _benchmark_ awal peforma dari sistem klasifikasi
- Memanfaatkan model ensemble tree yang unggul dalam _task_ klasifikasi dari beberapa fitur

## Data Understanding
Data yang digunakan adalah data **Machine Predictive Maintenance Classification** yang saya peroleh di kaggle memlalui link berikut ini:
https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification/data

Dataset tersebut berisi 10.000 data dengan 2 kolom ID, 6 kolom _Feature_, dan 2 kolom target.
- UID : unique identifier yang dimulai dari 1 hingga 10.000
- productID : Berisi jenis type dan nomor serial dari suatu mesin
- Type : Indikasi dari kualitas produk, L untuk low, M untuk medium, dan H untuk High
- air temperature : dihasilkan dengan menggunakan proses _random walk_ yang kemudian dinormalisasi ke standar deviasi 2 K sekitar 300 K
- process temperature : dihasilkan menggunakan proses _random walk_ yang dinormalisasi dengan standar deviasi 1 K, kemudian ditambahkan ke suhu udara ditambah 10 K
- rotational speed : dihitung dari daya sebesar 2860 W, dengan tambahan noise yang terdistribusi normal
- torque : nilai torsi terdistribusi normal di sekitar 40 Nm dengan σ = 10 Nm dan tidak ada nilai negatif
- tool wear : varian kualitas H/M/L menambah keausan alat masing-masing sebesar 5/3/2 menit pada alat yang digunakan dalam proses
- Target : Menentukan mesin gagal atau tidak
- Failure Type : Tipe kegagalan dari suatu mesin

### Data Loading
Pertama tentu kita harus melakukan load data yang akan kita gunakan
```python
df = pd.read_csv('/content/predictive_maintenance.csv')
```
Dengan code tersebut, dataset dengan format csv akan kita load dan ubah menjadi Data Frame. Setelah itu kita akan menampilkan sampel data frame yang sudah kita buat dari dataset yang kita gunakan.
```python
df.head()
```
Output:

![image](https://github.com/user-attachments/assets/30dd0e2a-10b9-4fc1-9e68-a2b1ff79693b)

### Data Info
Tipe data yang digunakan untuk setiap kolom akan di _breakdown_ menggunakan code berikut:
```python 
df.info()
```
Output:

![image](https://github.com/user-attachments/assets/f78e9338-0238-4295-a21b-5003470d024d)

Pada output tersebut dapat diketahui hampir seluruh fitur yang digunakan bertipe numeric yaitu int64 dan float64. Hanya ada kolom Product ID dan Type yang tipe datanya adalah Oject. Tidak sampai disitu, informasi soal keberadaan nilai null dapat kita ketahui memlalui jumlah data yang terbaca dan ternyata pada data ini tidak ada nilai null karena setiap kolom persis berjumlah 10.000 seluruhnya.

### Data Correlation
Sesuai dengan _Goals_ yang sudah ditentukan, kita perlu mencari tau korelasi antar fitur serta target yang ada pada dataset. Korelasi ini akan menghitung seberapa besar pengaruh suatu data dengan data yang lainnya.
```python
num_col = df.select_dtypes(include=['number']).columns

plt.figure(figsize = (15 , 10))
correlation_matrix = df[num_col].corr()
sns.heatmap(correlation_matrix , annot = True, cmap = 'coolwarm')
```
Pertama data yang akan kita cek korelasinya adalah data numerik saja. Oleh karena itu kita akan memilih data dengan angka didalamnya yang lalu akan didefine ke varibel bernama ```num_col```. Proses untuk menghitung korelasi akan dikerjakan oleh ```df[num_col].corr()``` dan akan divisualisasi menggunakan ```sns.heatmap```.

Output:

![image](https://github.com/user-attachments/assets/981d3d20-817e-4779-a803-b2c121cfc76c)

Jika kita amati, sebenarnya tidak ada fitur yang memiliki korelasi yang tinggi dengan **Target**, sebab tidak ada fitur yang memiliki nilai korelasi yang mendekati **angka 1**. Sehingga fitur-fitur yang ada pada dataset tersebut bisa dikategorikan memiliki korelasi yang lemah terhadap **Target**. Beberapa informasi yang dapat kita _extract_ dari _Correlation Matrix_ tersebut adalah sebagai berikut:
- Fitur memiliki korelasi yang lemah terhadap **Target**
- Fitur **Torque** memiliki korelasi paling tinggi dengan **Target**
- Fitur **Process Temperature** memiliki korelas paling renda terhadap **Target**
- Fitur **Rotational Speed** berkorelasi terbalik dengan **Target**

## Data Preparation
Sebelum melakukan _modeling_, bahan atau data yang akan kita masukan ke model tentu saja harus dilakukan _treatment_ agar dapat berjalan _seamless_ serta mendapatkan peforma terbaik saat diproses oleh model nantinya. 
### Target Validation
Seperti yang kita ketahui, Kolum **Target** hanya berisi angka **1** yang berarti **Failure** dan **0** yang berarti **No Failure**, lalu selain target ada juga kolom **Feature Type** yang mengkategorikan tipe dari **Failure** tersebut. Kecurigaan saya terhadap kondisi tersebut ialah, adanya kemungkinan terjadinya ketidaksesuian antara kolom **Target** dan **Feature Type**. Jika masi bingung kita akan masuk sekalian pada implementasi codenya untuk menghadapi tantangan ini.
```python
data1 = df[df.Target == 1]
data1['Failure Type'].value_counts()
```
Pada code tersebut, kita akan mencari data yang pada kolom **Target** nya bernilai 1 dan akan kita define dengan nama ```data1```. Setelah itu dengan ```value_counts()``` kita akan dapat melihat penjabarannya.
Output:

![image](https://github.com/user-attachments/assets/03e2af38-dae5-494a-9d4e-a48359348370)

Oke seperti dugaan awal, ternyata ada 9 data dengan kolom **Target** bernilai 1 yang memiliki nilai No Failure pada kolom **Feature Type**
Begitupun sebaliknya, kita juga akan _checking_ pada **Target** yang bernilai 0 menggunakan code yang mirip seperti diatas dan hasilnya adalah sebagai berikut:

![image](https://github.com/user-attachments/assets/658dba1c-9e66-4b6d-8c59-8fc51f890a76)

Dapat dilihat, data yang label **Target** nya 0 tetap dikategorikan memiliki tipe _failure_. 

Tentu saja hal ini akan membuat model kebingungan atau membuat hasil klasifikasi menjadi tidak kredibel dikarenakan adanya kesalahan pada data. Langkah paling cepat dan efektif tentu saja adalah dengan cara menghapus data yang salah tersebut mengingat jumlah data yang salah sangat minim dari jumlah total data yang kita miliki. Proses penghapusan akan menggunakan fungsi ```drop``` dengan memanfaatkan index data yang dikategorikan sebagai data yang salah pada Data Frame.
```python
index2 = data2[data2['Failure Type'] == 'Random Failures'].index
df.drop(index2 , axis =0 , inplace = True )
```
### Encoding Data
Pada data yang kita miliki dan yang akan kita jadikan fitur, terdapat satu kolom yang memiliki tipe data categorical yaitu **Type**. Oleh karena itu kita akan melakukan encoding untuk mengubahnya menjadi angka karena seperti yang kita tahu, komputer atau mesin hanya dapat memproses angka, hal tersebut juga berlaku pada model machine learning. Karena **Type** ini berisi categorical data, maka teknik _One Hot Encoding_ akan menjadi pilihan paling tepat. 
```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
data_encode = ohe.fit_transform(df[['Type']])

df_encode = pd.DataFrame(data_encode, columns=ohe.get_feature_names_out(['Type']), index=df.index)
df_fix = pd.concat([df, df_encode], axis=1)
df_fix.drop('Type', axis=1, inplace=True)
df_fix.head()
```
### Data Scaling
Mungkin ada yang berpikir bahwa proses ini melewatkan yang namanya outlier, tenang saja proses penanganan outlier ini akan kita selesaikan melaui scaling data. Oleh karena itu saya memilih Quantile Transformer dari library scikit learn sebagai teknik scaling datanya. Quantile Transformer sendiri bekerja dengan mengubah fitur-fitur agar mengikuti distribusi uniform (seragam) atau normal. Metode ini mampu untuk mengurangi dampak dari outlier pada fitur fitur yang digunakan. Sebenarnya untuk implementasinya sungguh sederhana, mari kita _dive_ ke codenya.
```python
df_fix.drop('Failure Type', axis=1, inplace=True)

X = df_fix.drop('Target', axis=1)
y = df_fix['Target']
```
Pertama kita harus membagi data yang kita gunakan menjadi dua kategori yaitu X dan y atau bisa kita sebut fitur dan target. Sesuai namanya pada variabel ``X`` kita _define_ fitur fitur yang akan kita gunakan, sementara variabel ``y``untuk targetnya. Baru setelah itu kita bisa melakukan scaling dengan Quantile Transformer
```python
from sklearn.preprocessing import QuantileTransformer

scaler = QuantileTransformer()
X_scaled = scaler.fit_transform(X)
```
Tentu saja, jangan lupa untuk melakukan import QuantileTransformer dari library sklearn. Kemudian melalui ``fit_transform``, Quantile Transformer akan diterapkan pada fitur.

## Modeling
Baiklah kita sampai pada tahap paling menarik bagi Machine Learning Engineer, tidak lain dan tidak bukan adalah tahap modeling. Seperti yang kita tahu, problem yang kita miliki akan kita selesaikan dengan metode klasifikasi. Oleh karena itu, kita akan mencoba berekperimen dengan beberapa model yang memiliki kemampuan dalam task klasifikasi. Rencananya saya akan menggunakan 1 model baseline dan 2 model ensemble dengan harapan dapat memberikan referensi yang lebih luas serta peforma yang baik. Ketiga algoritma yang akan saya gunakan diantara lain:
- Logistic Regression
- Random Forest
- XGBoost

### Logistic Regression
Algoritma paling sederhana yang cocok untuk klasifikasi adalah logistic regression. Logistic regression sendiri memiliki cara kerja memperkirakan suatu peristiwa atau sebuah kejadian berdasarkan data yang diberikan. Berikut adalah beberapa kelebihan yang dimiliki oleh Logistic Regression:
- **Kesederhanaan** : Model Logistic Regression merupakan salah satu model baseline yang paling sederhana implementasinya. Baik secara matematis pun, Logistic Regression tidak memiliki tingkat kerumitan setinggi algoritma lain.
- **Kecepatan** : Berkaitan dengan kesedehanaan sebelumnya, model logistic regression memiliki waktu _processing_ yang cukup cepat. Hal ini dikarenakan model yang sederhana ini tidak akan mengonsumsi daya komputasi yang terlalu besar berkat kesederhanaannya.

Berikut cara implementasinya:
```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train , y_train)
```
Sanagat sederhana bukan? Setelah proses import, kita hanya perlu melakukan fitting algoritma ke data latih yang kita miliki. 

### Random Forest
Algortima ensemble pertama yang akan kita gunakan adalah Random Forest, algoritma ensemble yang terbentuk dari beberapa pohon keputusan. Jumlah pohon keputusan yang akan terbentuk akan bergantung pada kompleksitas data dan jumlah fitur yang kita miliki. Hasil dari klasifikasi pada random forest akan ditentukan melalui sebuah sistem voting dari setiap pohon keputusannya. Berikut kelebihan yang dimiliki oleh Ra**ndom Forest:
- Kemampuan menangani fitur yang banyak** : Algoritma random forest sangat ahli dalam menangani data dengan banyak kolom atau banya fitur. 
- **Penanganan pada data yang tidak seimbang** : Random Forest dapat bekerja dengan baik pada data yang tidak seimbang dengan menyesuaikan bobot antar kelas.

Berikut cara implementasinya:
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
```
Cara implementasi yang mirip sebenarnya dengan logistic regression, tetapi disini ada penambahan pengaturan parameter ``random_state`` untuk mengatur tingkat keacakan dalam pemilihan data subset yang akan dimasukan ke setiap pohon keputusan.

### XGBoost
XGBoost atau _Extreme Gradient Boosting_ adalah algoritma yang memanfaatkan proses ensemble dari sejumlah pohon keputusan. Hanya saja penambahan pohon keputusan ini dilakukan secara bertahap. Proses _boosting_ dilakukan dengan cara meminimalkan loss function dengan memanfaatkan _gradient descent_. Beberapa kelebihan XGBoost diantara lain:
- **Peforma tinggi** : Gradient boosting membuat algoritma ini memiliki peforma yang sangat baik dalam menanagani data yang cukup kompleks terutama pada tugas klasifikasi.
- **Variasi parameter** : XGBoost sendiri memiliki banyak parameter yang dapat disesuaikan untuk mengoptimalkan kinerja dari algoritma ini.

Berikut cara implementasinya:
```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, y_train)
```
Parameter yang digunakan kali ini hanya ``random_state`` dan ``objective``. Seperti asumsi kita, karena klasifikasi yang akan kita lakukan ini hanya ada dua target yaitu '0' dan '1', maka objective dari klasifikasi ini adalah binary.

## Evaluasi Model
Tidak lengkap jika kita tidak melakukan penilaian atau mengukur peforma model kita setelah kita melatihnya. Proses evaluasi ini ditunjukan untuk melihat seberapa baik model kita memproses data uji dengan berbagai macam metrik evaluasi. Metrik evaluasi yang akan kita gunakan adalah ``f1-score``, metrik evaluasi yang paling umum digunakan untuk _task_ klasifikasi. Sebelum itu kita perlu mengenal yang namanya precision dan recall, precision adalah ukuran akurasi dari prediksi positif yang dibuat oleh model. Dengan kata lain precision akan menghitung jumlah prediksi positif yag benar terhadap jumlah keseluruhan prediksi positif yang dilakukan oleh model. Sementara recall adalah ukuran seberapa baik model menangkap semua instance positif di dalam data. Maka recall ini bisa sebagai tingkat prediksi positif dari kesekuruhan prediksi positif yang seharusnya ada. F-1 score hadir apabila kita menginginkan keseimbangan antara precision dan recall. Langsung saja kita _jump-in_ menuju implementasinya
```python
from sklearn.metrics import classification_report

y_pred_lr = clf.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

print("Logistic Regression:")
print(classification_report(y_test, y_pred_lr))

print("\nXGBoost:")
print(classification_report(y_test, y_pred_xgb))

print("\nRandom Forest:")
print(classification_report(y_test, y_pred_rf))
```
Pada code tersebut, kita akan menggunakan classification report yang berupa penjabaran evaluasi peforma dari model yang digunakan.
Output:

![image](https://github.com/user-attachments/assets/5d938727-a048-4b8c-bcbc-83c137cadf68)

Berdasarkan hasil yang kita dapat, nampaknya peforma dari ketiga model tersebut tidak terlalu jauh jika kita hanya berpatokan pada ``f1-score`` keseluruhan. Namun seperti yang bisa kita lihat, model logistic regression sangat buruk dalam memprediksi kelas yang minoritas terbukti dengan nilai ``f1-score`` yang amat kecil pada kelas '1'. Sementara XGBoost memiliki peforma yang paling baik diantara dua model lainnya jika kita melihat nilai ``f1-score`` nya di dua kelas yang ada. Pada kasus melalukan prediksi apakah mesin failure atau tidak, nilai ``recall`` lebih penting dibandingkan dengan ``precision``. Kesimpulan tersebut diambil karena akan lebih baik jika kita mendeteksi setiap mesin yang rusak ketimbang kita melewatkan banyak mesin yang rusak dan memprediksi mesin yang normal sebagai rusak. Karena pada dasarnya maintenance itu adalah hal yang baik dan tidak akan merugikan kinerja mesin tersebur. Maka dari itu algoritma XGBoost yang memiliki tingkat ``f1-score`` dan ``recall`` yang paling tinggi akan menjadi model unggulan pada permasalahan ini.
