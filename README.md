## Laporan-Proyek-Machine-Learning-Hansen-Jonathan

#Domain Proyek
        Sejak 1991, Indonesia menyatakan mempunyai 718 bahasa lokal. Badan Pengembangan dan Pembinaan Bahasa melakukan studi terhadap 94 bahasa lokal untuk melihat kehidupan dari bahasa tersebut. Ternyata sudah terdapat 5 bahasa yang punah dan sisanya juga sudah mulai sedikit. Hal ini membuat Nadiem Makarim selaku Menteri Pendidikan Indonesia menyatakan bahwa bahasa daerah di Indonesia terancam punah. Karena banyak orang sudah melupakan bahasa-bahasa daerah terlebih penulisan-penulisan aksara. Sehingga melalui proyek ini, diharapkan orang-orang bisa menulis dan belajar sendiri aksara Sunda dengan tepat. Penelitian serupa untuk melestarikan bahasa ini juga sudah pernah dilakukan sebelumnya oleh Febriansyah, Feggy, et al. "Pengenalan Teknologi Android Game Edukasi Belajar Aksara Sunda untuk Meningkatkan Pengetahuan." JURIKOM (Jurnal Riset Komputer) 8.6 (2021): 336-344.


#Business Understanding
1. Problem Statement:
   Bagaimana supaya orang dapat menulis dan belajar sendiri aksara Sunda dengan tepat?
2. Goals:
   Menciptakan machine learning model yang mampu melakukan klasifikasi penulisan aksara Sunda dengan tepat


#Data Understanding
Data yang digunakan berasal dari https://github.com/ridhomujizat/AksaraSundaCNN/.
Data Loading: jika kita melihat pada data lokal, terdapat 18 kelas masing-masing pada data train dan test.


#Data Preparation

        Data train / data latih akan digunakan untuk melatih model, sedangkan data test akan digunakan sebagai validation untuk model. Pada data train akan dilakukan normalisasi dengan membagi (rescale) semuanya dengan 1/255. Lalu gambar juga akan dilakukan augmentasi. Tetapi untuk data validation hanya dilakukan normalisasi saja. Lalu tiap data train dan validation akan diubah warnanya biar sama. Dengan ImageDataGenerator kita dapat memproduksi berbagai varisasi data tanpa memakan atau menggunakan 'space' penyimpanan kita, sehingga model dapat lebih belajar banyak variasi data, seperti foto pada gambar diperbesar, dibalik secara horizontal maupun vertikal, tetapi karena ini tulisan aksara kita tidak menggunakan flip karena artinya akan berbeda nanti.


#Modeling
        Pada awal layer, kita menggunakan conv2D untuk membentuk lapisan konvolusi karena data yang kita masukkan berupa tensor 2 dimensi kita menggunakan aktivasi relu atau Rectified Linear Unit karena keuntungannya yaitu mempercepat proses konvergensi yang dilakukan dengan stochastic gradient descent jika dibandingkan dengan sigmoid / tanh dan padding agar semua memiliki ukuran yang sama, tidak lupa juga untuk lapisan pertama argumen input_shape, (112,112,1) karena gambar input kita memiliki ukuran 100x100 dan warna hanya hitam putih jadi kita menulisnya 100x100x1. Lalu kita menulis lapisan berikutnya seperti LeakyReLU, bisa dilakukan pemanggilan atau aktivasi layer lain, hasil ini diperoleh dari trial and error. Lalu MaxPooling2D untuk operasi pooling. Lalu GlobalMaxPool2D lalu Dense, pada akhir layer digunakan activation softmax karena klasifikasi yang kita lakukan lebih dari 2 dan output 18 karena kita mempunyai 18 kelas data. Untuk model digunakan optimasi Adam, karena label kita akan berupa one-hot-encoded, kita menggunakan categorical_crossentropy, lalu dengan metrik yang melakukan judge pada model melihat dari akurasi yang dihasilkan.


#Evaluation

![plot acc and val acc](https://user-images.githubusercontent.com/106476815/180648567-5635fd26-3029-4b01-a665-aaf4426e5338.png)
![plot loss](https://user-images.githubusercontent.com/106476815/180648570-425c8f9b-1add-47ff-a513-d1b74a0036ff.png)

Dari hasil plot setelah model dilatih kita melihat bahwa hasil model masih harus ditingkatkan kembali karena validation accuracy dengan training accuracy masih terjadi jarak (gap) yang sangat jauh. Jika dilihat dari training loss dan validation loss pada plot juga butuh tuning up agar model lebih baik.
