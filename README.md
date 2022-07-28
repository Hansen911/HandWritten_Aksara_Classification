# Laporan-Proyek-Machine-Learning-Hansen-Jonathan

## Domain Proyek
Pesatnya perkembangan teknologi mencakup banyak bidang yang saling berhubungan satu sama lain terutama dalam bidang pendidikan dimana ketika suatu teknologi diterapkan proses pembelajaran tidak hanya sebatas antara guru dan siswa tetapi siswa dapat mengakses pembelajaran melalui berbagai media. Kurangnya sistem pembelajaran yang masih menggunakan metode tradisional sehingga menimbulkan kesan bosan. Dari permasalahan yang terjadi menyebabkan kurangnya pengetahuan anak-anak tentang aksara sunda, karena kurangnya pengetahuan dalam mengenal aksara sunda dapat mengakibatkan hilangnya bahasa daerah di Indonesia karena terlalu banyaknya unsur bahasa dan budaya dari luar[1].Sejak 1991, Indonesia menyatakan mempunyai 718 bahasa lokal. Badan Pengembangan dan Pembinaan Bahasa melakukan studi terhadap 94 bahasa lokal untuk melihat kehidupan dari bahasa tersebut. Ternyata sudah terdapat 5 bahasa yang punah dan sisanya juga sudah mulai sedikit. Hal ini membuat Nadiem Makarim selaku Menteri Pendidikan Indonesia menyatakan bahwa bahasa daerah di Indonesia terancam punah. Karena banyak orang sudah melupakan bahasa-bahasa daerah terlebih penulisan-penulisan aksara. Sehingga melalui proyek ini, diharapkan orang-orang bisa menulis dan belajar sendiri aksara Sunda dengan tepat.


## Business Understanding
1. Problem Statement:
   Bagaimana supaya orang dapat menulis dan belajar sendiri aksara Sunda dengan tepat?
2. Goals:
   Menciptakan machine learning model yang mampu melakukan klasifikasi penulisan aksara Sunda dengan tepat. Dengan klasifikasi tulisan yang pelajar tulis, ia dapat memahami apakah yang ia tulis sudah benar atau belum. Sehingga dengan hal ini, pelajar dapat belajar sendiri penulisan aksara Sunda dengan tepat


## Data Understanding
Data yang digunakan berasal dari https://github.com/ridhomujizat/AksaraSundaCNN/.
Data Loading: jika kita melihat pada data lokal, terdapat 18 kelas masing-masing pada data train dan test.

![gambar fitur](https://user-images.githubusercontent.com/106476815/181236794-5a64370f-263c-4633-a23f-a90f0d87b3db.png)
ba
ca
da
ga
ha
ja
ka
la  
ma
na
nga
nya
pa
ra 
sa
ta
wa 
ya

((354, 354),
array([[1., 1., 1., ..., 1., 1., 1.],
        [1., 1., 1., ..., 1., 1., 1.],
        [1., 1., 1., ..., 1., 1., 1.],
        ...,
        [1., 1., 1., ..., 1., 1., 1.],
        [1., 1., 1., ..., 1., 1., 1.],
        [1., 1., 1., ..., 1., 1., 1.]]))
        
Pada gambar awal, gambar berukuran 354x354 dan hitam putih. Fitur ekstrak ini menghasilkan sebuah angka dimana setiap angka merepresentasikan sebuah warna.


## Data Preparation

Data train / data latih akan digunakan untuk melatih model, sedangkan data test akan digunakan sebagai validation untuk model. Pada data train akan dilakukan normalisasi dengan membagi (rescale) semuanya dengan 1/255. Lalu gambar juga akan dilakukan augmentasi. Tetapi untuk data validation hanya dilakukan normalisasi saja. Lalu tiap data train dan validation akan diubah warnanya biar sama. Dengan ImageDataGenerator kita dapat memproduksi berbagai varisasi data tanpa memakan atau menggunakan 'space' penyimpanan kita, sehingga model dapat lebih belajar banyak variasi data, seperti foto pada gambar diperbesar, dibalik secara horizontal maupun vertikal, tetapi karena ini tulisan aksara kita tidak menggunakan flip karena artinya akan berbeda nanti.


## Modeling
Pada awal layer, kita menggunakan conv2D untuk membentuk lapisan konvolusi karena data yang kita masukkan berupa tensor 2 dimensi kita menggunakan aktivasi relu atau Rectified Linear Unit karena keuntungannya yaitu mempercepat proses konvergensi yang dilakukan dengan stochastic gradient descent jika dibandingkan dengan sigmoid / tanh dan padding agar semua memiliki ukuran yang sama, tidak lupa juga untuk lapisan pertama argumen input_shape, (112,112,1) karena gambar input kita memiliki ukuran 100x100 dan warna hanya hitam putih jadi kita menulisnya 100x100x1. Lalu kita menulis lapisan berikutnya seperti LeakyReLU, bisa dilakukan pemanggilan atau aktivasi layer lain, hasil ini diperoleh dari trial and error. Lalu MaxPooling2D untuk operasi pooling. Lalu GlobalMaxPool2D lalu Dense, pada akhir layer digunakan activation softmax karena klasifikasi yang kita lakukan lebih dari 2 dan output 18 karena kita mempunyai 18 kelas data. Untuk model digunakan optimasi Adam, karena label kita akan berupa one-hot-encoded, kita menggunakan categorical_crossentropy, lalu dengan metrik yang melakukan judge pada model melihat dari akurasi yang dihasilkan.


## Evaluation

![plot acc and val acc](https://user-images.githubusercontent.com/106476815/180648567-5635fd26-3029-4b01-a665-aaf4426e5338.png)
![plot loss](https://user-images.githubusercontent.com/106476815/180648570-425c8f9b-1add-47ff-a513-d1b74a0036ff.png)

loss: 0.0183 - accuracy: 0.9955 - val_loss: 1.5512 - val_accuracy: 0.6846
loss adalah nilai yang didapat dari hasil model melakukan training menggunakan data training sedangkan val_loss adalah nilai yang didapat dari hasil model menggunakan data validasi. Keduanya memiliki arti yang sama yaitu menilai seberapa buruk model memprediksi suatu hal, semakin baik model maka nilai loss dan val_loss akan bernilai makin kecil atau bahkan mendekati 0.  
accuracy merupakan nilai dari hasil model yang dilatih menggunakan data latih, sedangkan val_accuracy merupakan nilai dari hasil model memprediksi sampel yang tidak ikut terlatih atau kita bisa sebut seberapa besar akurasi model jika digunakan pada kasus nyata.
Pada klasifikasi gambar kita menggunakan nilai accuracy sebagai metrik, kita mendapatkan hasil yang tinggi. Sehingga model dapat dikatakan sudah bagus, tetapi kita bisa lihat dan bandingkan accuracy dengan val_accuracy yang cukup jauh berbeda, begitu juga dengan nilai val_loss yang jika dilihat pada grafik belum monoton turun, dari sini bisa kita simpulkan bahwa model masih dapat kita tingkatkan untuk menjadi lebih baik.
Dengan model ini, pelajar dapat mengetahui apakah tulisan aksara yang ia tulis sudah tepat atau belum.

## References
[1]Febriansyah, Feggy, et al. "Pengenalan Teknologi Android Game Edukasi Belajar Aksara Sunda untuk Meningkatkan Pengetahuan." JURIKOM (Jurnal Riset Komputer) 8.6 (2021): 336-344.
