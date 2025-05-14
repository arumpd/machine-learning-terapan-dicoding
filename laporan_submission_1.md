# Laporan Proyek Machine Learning - Arum Puspadewi

## Domain Proyek

Sudah tidak dapat dimungkiri lagi bahwa pembatalan reservasi oleh customer berdampak kerugian dalam industri penginapan. Dengan meningkatnya jumlah pemesanan online, penting untuk memprediksi apakah reservasi akan dibatalkan, agar pihak hotel bisa mengambil langkah proaktif seperti overbooking atau promosi ulang.

**Rubrik/Kriteria Tambahan (Opsional)**:
- Jelaskan mengapa dan bagaimana masalah tersebut harus diselesaikan
- Menyertakan hasil riset terkait atau referensi. Referensi yang diberikan harus berasal dari sumber yang kredibel dan author yang jelas.
- Format Referensi dapat mengacu pada penulisan sitasi [IEEE](https://journals.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/7/IEEE_Reference_Guide.pdf), [APA](https://www.mendeley.com/guides/apa-citation-guide/) atau secara umum seperti [di sini](https://penerbitdeepublish.com/menulis-buku-membuat-sitasi-dengan-mudah/)
- Sumber yang bisa digunakan [Scholar](https://scholar.google.com/)

## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

- Apakah pelanggan akan membatalkan reservasi hotelnya?
- Fitur apa yang paling memengaruhi pembatalan reservasi?
- Bagaimana cara meningkatkan akurasi prediksi pembatalan?

### Goals

- Memprediksi probabilitas pembatalan pemesanan.
- Mengidentifikasi fitur paling berpengaruh dalam keputusan pembatalan.
- Menemukan model prediktif terbaik.

### Solution Statements

- Melakukan prediksi dengan algoritma machine learning.
- Mengukur performa dengan akurasi, precision, recall, dan F1-score.
- Menggunakan beragam algoritma (Random Forest, ANN, dan XGBoost).

## Data Understanding
Dataset ini berisi 119390 data berupa informasi pemesanan untuk hotel kota dan hotel resor, dan mencakup informasi seperti waktu pemesanan, lama menginap, jumlah orang dewasa, anak-anak, dan/atau bayi, serta jumlah tempat parkir yang tersedia. [Hotel booking demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand). 

### Variabel-variabel pada Hotel booking demand dataset adalah sebagai berikut:
- hotel : Jenis Penginapan (H1 = Hotel Resor or H2 = Hotel Kota)
- is_canceled : Nilai yang menunjukkan apakah pemesanan dibatalkan (1) atau tidak (0)
- lead_time : Jumlah hari yang telah berlalu antara tanggal masuknya pemesanan ke dalam PMS dan tanggal kedatangan
- arrival_date_year : Tahun dari tanggal kedatangan
- arrival_date_month : Bulan dari tanggal kedatangan
- arrival_date_week_number : Minggu dari tanggal kedatangan
- arrival_date_day_of_month : Hari dari tanggal kedatangan
- stays_in_weekend_nights : Jumlah malam akhir pekan (Sabtu atau Minggu) tamu menginap atau memesan untuk menginap di hotel
- stays_in_week_nights : Jumlah malam dalam seminggu (Senin hingga Jumat) tamu menginap atau memesan untuk menginap di hotel
- adults : Jumlah orang dewasa
- children : Jumlah anak-anak
- babies : Jumlah bayi
- meal : Jenis makanan yang dipesan. Kategori disajikan dalam paket makanan standar perhotelan: Tidak ditentukan/SC - tidak ada paket makanan; BB - Tempat Tidur & Sarapan; HB - Setengah harga (sarapan dan satu kali makan lainnya - biasanya makan malam); FB - Harga penuh (sarapan, makan siang, dan makan malam)
- country : Negara asal. Kategori diwakili dalam format ISO 3155-3:2013
- market_segment : Penunjukan segmen pasar. Dalam kategori, istilah “TA” berarti “Agen Perjalanan” dan ‘TO’ berarti “Operator Tur”
- distribution_channel : Saluran distribusi pemesanan. Istilah “TA” berarti “Agen Perjalanan” dan ‘TO’ berarti “Operator Tur”
- is_repeated_guest : Nilai yang menunjukkan apakah nama pemesanan berasal dari tamu berulang (1) atau tidak (0)
- previous_cancellations : Jumlah pemesanan sebelumnya yang dibatalkan oleh pelanggan sebelum pemesanan saat ini
- previous_bookings_not_canceled : Jumlah pemesanan sebelumnya yang tidak dibatalkan oleh pelanggan sebelum pemesanan saat ini
- reserved_room_type : Kode jenis kamar yang dipesan. Kode disajikan sebagai pengganti sebutan untuk alasan anonimitas.
- assigned_room_type : Kode untuk jenis kamar yang ditetapkan untuk pemesanan. Terkadang jenis kamar yang ditetapkan berbeda dengan jenis kamar yang dipesan karena alasan operasional hotel (misalnya pemesanan berlebih) atau atas permintaan pelanggan. Kode disajikan sebagai pengganti sebutan untuk alasan anonimitas.
- booking_changes : Jumlah perubahan/amandemen yang dilakukan pada pemesanan sejak pemesanan dimasukkan pada PMS hingga saat check-in atau pembatalan
- deposit_type : Indikasi apakah pelanggan melakukan deposit untuk menjamin pemesanan. Variabel ini dapat memiliki tiga kategori: Tanpa Deposit - tidak ada deposit yang dilakukan; Non Refund - deposit yang dilakukan senilai total biaya menginap; Dapat Dikembalikan - deposit yang dilakukan dengan nilai di bawah total biaya menginap.
- agent : ID biro perjalanan yang melakukan pemesanan
- company : ID perusahaan/entitas yang melakukan pemesanan atau yang bertanggung jawab untuk membayar pemesanan. ID ditampilkan sebagai pengganti sebutan untuk alasan anonimitas
- days_in_waiting_list : Jumlah hari pemesanan berada dalam daftar tunggu sebelum dikonfirmasi kepada pelanggan
- customer_type : Jenis pemesanan, dengan asumsi salah satu dari empat kategori: Kontrak - ketika pemesanan memiliki jatah atau jenis kontrak lain yang terkait dengannya; Grup - ketika pemesanan terkait dengan grup; Transien - ketika pemesanan bukan bagian dari grup atau kontrak, dan tidak terkait dengan pemesanan transien lainnya; Transien-pihak - ketika pemesanan bersifat transien, tetapi terkait dengan setidaknya pemesanan transien lainnya
- adr : Tarif Harian Rata-Rata yang ditentukan dengan membagi jumlah semua transaksi penginapan dengan jumlah total malam menginap
- required_car_parking_spaces : Jumlah ruang parkir mobil yang dibutuhkan oleh pelanggan
- total_of_special_requests : Jumlah permintaan khusus yang dibuat oleh pelanggan (misalnya tempat tidur kembar atau lantai tinggi)
- reservation_status : Status terakhir pemesanan, dengan asumsi salah satu dari tiga kategori: Dibatalkan - pemesanan dibatalkan oleh pelanggan; Check-Out - pelanggan telah melakukan check-in namun sudah berangkat; Tidak Datang - pelanggan tidak melakukan check-in dan menginformasikan alasannya kepada hotel
- reservation_status_date : Tanggal saat status terakhir ditetapkan. Variabel ini dapat digunakan bersama dengan ReservationStatus untuk memahami kapan pemesanan dibatalkan atau kapan pelanggan check-out dari hotel

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

### Exploratory Data Analysis

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
