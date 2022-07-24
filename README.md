# Laporan-Proyek-Machine-Learning---Hansen-Jonathan

Domain Proyek
        Banyak orang sudah melupakan bahasa-bahasa daerah terlebih penulisan-penulisan aksara. Sehingga melalui proyek ini, diharapkan orang-orang bisa menulis aksara Sunda dengan tepat. 


Business Understanding
1. Problem Statement:
   1. Punahnya bahasa-bahasa daerah di Indonesia
   2. Anak muda zaman sekarang tidak diajarkan bahasa daerahnya masing-masing
2. Goals:
   1. Bahasa daerah tetap dilestarikan
   2. Anak-anak dapat dengan mudah belajar bahasa daerah


Data Understanding
        Data yang digunakan berasal dari https://github.com/ridhomujizat/AksaraSundaCNN/. Terdapat 18 kelas masing-masing pada data train dan test.


Data Preparation
        Data train / data latih akan digunakan untuk melatih model, sedangkan data test akan digunakan sebagai validation untuk model. Pada data train akan dilakukan normalisasi dengan membagi (rescale) semuanya dengan 1/255. Lalu gambar juga akan dilakukan augmentasi. Tetapi untuk data validation hanya dilakukan normalisasi saja. Lalu tiap data train dan validation akan diubah warnanya biar sama.


      tf.keras.layers.Conv2D(256, (5,5), activation = 'relu', padding = 'same', input_shape=(100,100,1)),
      tf.keras.layers.LeakyReLU(alpha=0.2),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(256, (5,5)),
      tf.keras.layers.LeakyReLU(alpha=0.3),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Conv2D(256, (5,5)),
      tf.keras.layers.LeakyReLU(alpha=0.3),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.GlobalMaxPool2D(),
      tf.keras.layers.Dense(800),
      tf.keras.layers.LeakyReLU(alpha=0.2),
      tf.keras.layers.Dense(18, activation = 'softmax')



  model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics = ['accuracy'])
        


Layer (type)                Output Shape              Param #   
=================================================================
 conv2d_3 (Conv2D)           (None, 100, 100, 256)     6656      
                                                                 
 leaky_re_lu_4 (LeakyReLU)   (None, 100, 100, 256)     0         
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 50, 50, 256)      0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 46, 46, 256)       1638656   
                                                                 
 leaky_re_lu_5 (LeakyReLU)   (None, 46, 46, 256)       0         
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 23, 23, 256)      0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 19, 19, 256)       1638656   
                                                                 
 leaky_re_lu_6 (LeakyReLU)   (None, 19, 19, 256)       0         
                                                                 
 max_pooling2d_5 (MaxPooling  (None, 9, 9, 256)        0         
 2D)                                                             
                                                                 
 dropout_1 (Dropout)         (None, 9, 9, 256)         0         
                                                                 
 global_max_pooling2d_1 (Glo  (None, 256)              0         
 balMaxPooling2D)                                                
                                                                 
 dense_2 (Dense)             (None, 800)               205600    
                                                                 
 leaky_re_lu_7 (LeakyReLU)   (None, 800)               0         
                                                                 
 dense_3 (Dense)             (None, 18)                14418     
                                                                 
=================================================================
Total params: 3,503,986
Trainable params: 3,503,986
Non-trainable params: 0
_________________________________________________________________


        Pada layer awal, dibuat input sesuai dengan size gambar. Lalu digunakan layer max pooling, dropout dan relu juga supaya model lebih dapat mengekstrak informasi lebih tepat. Pada akhir layer digunakan activation softmax dan output 18 karena kita mempunyai 18 kelas data. Untuk model digunakan optimasi Adam, karena label kita akan berupa one-hot-encoded, kita menggunakan categorical_crossentropy, lalu dengan metrik yang melakukan judge pada model melihat dari akurasi yang dihasilkan.


Evaluation
  

  

        Dari hasil plot setelah model dilatih kita melihat bahwa hasil model masih harus ditingkatkan kembali karena validation accuracy dengan training accuracy masih terjadi jarak (gap) yang sangat jauh. Jika dilihat dari training loss dan validation loss pada plot juga butuh tuning up agar model lebih baik.
