## Laporan-Proyek-Machine-Learning-Hansen-Jonathan

#Domain Proyek
        Sejak 1991, Indonesia menyatakan mempunyai 718 bahasa lokal. Badan Pengembangan dan Pembinaan Bahasa melakukan studi terhadap 94 bahasa lokal untuk melihat kehidupan dari bahasa tersebut. Ternyata sudah terdapat 5 bahasa yang punah dan sisanya juga sudah mulai sedikit. Hal ini membuat Nadiem Makarim selaku Menteri Pendidikan Indonesia menyatakan bahwa bahasa daerah di Indonesia terancam punah. Karena banyak orang sudah melupakan bahasa-bahasa daerah terlebih penulisan-penulisan aksara. Sehingga melalui proyek ini, diharapkan orang-orang bisa menulis aksara Sunda dengan tepat.


#Business Understanding
1. Problem Statement:
   a. Bagaimana cara melestarikan bahasa daerah?
   b. Mengapa tidak ada anak-anak yang belajar bahasa daerah?
2. Goals:
   a. Dengan menciptakan suatu pembelajaran yang unik sehingga banyak orang mulai tertarik dan kembali belajar dan juga memperkenalkan kembali kepada masyarakat umum.
   b. Akses belajar bahasa daerah yang juga tidak mudah dan terkadang berbayar membuat beberapa kalangan tidak dapat belajar.


#Data Understanding
Data yang digunakan berasal dari https://github.com/ridhomujizat/AksaraSundaCNN/.
Data Loading: jika kita melihat pada data lokal, terdapat 18 kelas masing-masing pada data train dan test.


#Data Preparation

Do the image augmentation and rescale image
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   width_shift_range = 0.2, height_shift_range = 0.2,
                                   shear_range = 0.2, zoom_range = 0.2, fill_mode='nearest')

train_data = train_datagen.flow_from_directory(
    directory=root_path_train,
    color_mode="grayscale",
    batch_size=30)

validation_datagen = ImageDataGenerator(rescale = 1./255)

validation_data = validation_datagen.flow_from_directory(
    directory=root_path_test,
    target_size=(112,112),
    color_mode="grayscale",
    batch_size=30)
    

        Data train / data latih akan digunakan untuk melatih model, sedangkan data test akan digunakan sebagai validation untuk model. Pada data train akan dilakukan normalisasi dengan membagi (rescale) semuanya dengan 1/255. Lalu gambar juga akan dilakukan augmentasi. Tetapi untuk data validation hanya dilakukan normalisasi saja. Lalu tiap data train dan validation akan diubah warnanya biar sama.


#Modeling


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


        Pada layer awal, dibuat input sesuai dengan size gambar. Lalu digunakan layer max pooling, dropout dan relu juga supaya model lebih dapat mengekstrak informasi lebih tepat. Pada akhir layer digunakan activation softmax karena klasifikasi yang kita lakukan lebih dari 2 dan output 18 karena kita mempunyai 18 kelas data. Untuk model digunakan optimasi Adam, karena label kita akan berupa one-hot-encoded, kita menggunakan categorical_crossentropy, lalu dengan metrik yang melakukan judge pada model melihat dari akurasi yang dihasilkan.


#Evaluation

![plot acc and val acc](https://user-images.githubusercontent.com/106476815/180648567-5635fd26-3029-4b01-a665-aaf4426e5338.png)
![plot loss](https://user-images.githubusercontent.com/106476815/180648570-425c8f9b-1add-47ff-a513-d1b74a0036ff.png)

Dari hasil plot setelah model dilatih kita melihat bahwa hasil model masih harus ditingkatkan kembali karena validation accuracy dengan training accuracy masih terjadi jarak (gap) yang sangat jauh. Jika dilihat dari training loss dan validation loss pada plot juga butuh tuning up agar model lebih baik.
