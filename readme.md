<h1 align="center">Monkeypox Skin Lesion Detection Using Deep Learning</h1>

## Introduction
Kasus MonkeyPox dilaporkan meningkat tajam di berbagai wilayah, termasuk Afrika, Eropa, Amerika, dan Asia. Penyebaran penyakit ini memunculkan kekhawatiran bahwa MonkeyPox dapat berkembang menjadi pandemi global seperti Covid-19 pada tahun 2020. Kasus Mpox yang terus meningkat di berbagai negara menghadirkan tantangan besar, terutama dalam melakukan diagnosis dini. Untuk mencegah penyebaran lebih luas, sangat penting untuk mengenali pembawa (carrier) penyakit ini sejak dini agar tidak menularkan kepada orang lain di sekitarnya. Salah satu upaya yang dapat dilakukan adalah dengan memanfaatkan teknologi penginderaan jarak jauh, seperti kamera berkualitas tinggi, untuk mendeteksi tanda-tanda MonkeyPox pada kulit seseorang.  Untuk mengetahui apakah seseorang terinfeksi Mpox, diperlukan ahli yang dapat mengenali perubahan pada kulit sebagai gejala penyakit ini. Namun, identifikasi ini membutuhkan ahli yang mampu memastikan apakah perubahan pada kulit tersebut merupakan gejala MonkeyPox atau bukan. Namun, jumlah ahli yang terbatas membuat mereka sulit memantau dan mendiagnosis setiap kasus secara berkelanjutan. Oleh karena itu, diperlukan sebuah sistem yang dapat mengklasifikasikan ciri-ciri kulit yang sehat, MonkeyPox ataupun penyakit lain dengan **efisien, presisi, dan akurasi yang baik.** Diberikan dataset yang dapat di akses pada link berikut: [Source](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20/data)

## Data
Dataset terbagi menjadi dua kategori, yaitu:
#### 1. Original Images
Original images terdiri dari 5 folds (lipatan) yang di dalamnya terdapat folder Train, Valid, dan Test. Masing-masing berisikan citra dari kulit dengan 6 kelas berbeda.

#### 2. Augmented Images
Augmented images terdiri dari 5 folds (lipatan) yang di dalamnya hanya terdapat folder Train untuk masing-masing kelas.