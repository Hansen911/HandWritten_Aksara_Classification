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
18 kelas tersebut terdiri dari:
ba = gambar aksara sunda huruf ba,
ca = gambar aksara sunda huruf ca,
da = gambar aksara sunda huruf da,
ga = gambar aksara sunda huruf ga,
ha = gambar aksara sunda huruf ha,
ja = gambar aksara sunda huruf ja,
ka = gambar aksara sunda huruf ka,
la = gambar aksara sunda huruf la,
ma = gambar aksara sunda huruf ma,
na = gambar aksara sunda huruf na,
nga = gambar aksara sunda huruf nga,
nya = gambar aksara sunda huruf nya,
pa = gambar aksara sunda huruf pa,
ra = gambar aksara sunda huruf ra,
sa = gambar aksara sunda huruf sa,
ta = gambar aksara sunda huruf ta,
wa = gambar aksara sunda huruf wa,
ya = gambar aksara sunda huruf ya,

berikut adalah contoh gambar aksara sunda huruf ba

![gambar fitur](https://user-images.githubusercontent.com/106476815/181236794-5a64370f-263c-4633-a23f-a90f0d87b3db.png)

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

![acc](https://user-images.githubusercontent.com/106476815/181580647-3aa65748-9514-4764-9b2f-71c43ae9e660.png)
![loss](https://user-images.githubusercontent.com/106476815/181580636-831aaab5-6d09-4519-a4c0-422c95f0db15.png)

loss: 0.0179 - accuracy: 0.9944 - val_loss: 0.0032 - val_accuracy: 0.9988
loss adalah nilai yang didapat dari hasil model melakukan training menggunakan data training sedangkan val_loss adalah nilai yang didapat dari hasil model menggunakan data validasi. Keduanya memiliki arti yang sama yaitu menilai seberapa buruk model memprediksi suatu hal, semakin baik model maka nilai loss dan val_loss akan bernilai makin kecil atau bahkan mendekati 0.  
accuracy merupakan nilai dari hasil model yang dilatih menggunakan data latih, sedangkan val_accuracy merupakan nilai dari hasil model memprediksi sampel yang tidak ikut terlatih atau kita bisa sebut seberapa besar akurasi model jika digunakan pada kasus nyata.

![metric](https://user-images.githubusercontent.com/106476815/181580642-909dc65a-b98c-4b11-afcb-7112a78aed4b.png)

Pada klasifikasi gambar kita menggunakan nilai accuracy sebagai metrik, kita mendapatkan hasil yang tinggi.
Dengan model ini, pelajar dapat mengetahui apakah tulisan aksara yang ia tulis sudah tepat atau belum.

## References
[1]Febriansyah, Feggy, et al. "Pengenalan Teknologi Android Game Edukasi Belajar Aksara Sunda untuk Meningkatkan Pengetahuan." JURIKOM (Jurnal Riset Komputer) 8.6 (2021): 336-344.
