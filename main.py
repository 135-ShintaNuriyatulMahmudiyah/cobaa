import streamlit as st
import pandas as pd
import numpy as np
import pip
pip.main(["install", "openpyxl"])
import re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
pip.main(["install", "Sastrawi"])
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


st.title('Analisis Sentimen - Web APP')
st.text("""---------------------------------------------------------------------------------------------""")
st.title(""" Analisis Sentimen Ulasan Pengunjung Wisata Di Pulau Madura Menggunakan Metode Logistic Regression Dengan Seleksi Fitur Chi–Squere""")
st.text("""Dibuat Oleh : Shinta Nuriyatul Mahmudiyah - 200411100135""")
st.text("""---------------------------------------------------------------------------------------------""")
Home, Learn, Proses, Model, Visualisasi,Implementasi = st.tabs(['Home', 'Learn Data', 'Preprocessing ', 'TF-IDF dan Model','Visualisasi', 'Implementasi'])
with Home:
    st.title("""
    Menggunakan Dataset yang berbeda dengan metode Logistic Regression dan seleksi fitur Chi - Squere. 
    Mana akurasi yang terbaik?
    """)
    st.write('Pariwisata merupakan salah satu sektor yang dapat meningkatkan perekonomian suatu daerah. Salah satunya Pulau Madura ini yang mempunyai banyak keindahan alam yang belum banyak masyarakat mengetahui wisatanya dan pengembangan wisata sangatlah diperlukan untuk menarik wisatawan untuk bisa menikmati wisata di di Pulau Madura.')
    st.write('Dalam Klasifikasi ini data yang digunakan adalah ulasan atau komentar dari Google maps dengan bantuan extension Instan Data Scrapper.')
    st.write('Analisis sentimen merupakan cara yang digunakan untuk memahami pandangan dan perasaan wisatawan terhadap sebuah destinasi wisata. Penelitian ini bertujuan untuk mengetahui nilai akurasi yang didapatkan analisis sentimen    dalam pengklasifian sentimen   positif dan negatif  terhadap ulasan pengunjung wisata di Pulau Madura dengan metode logistic regression dengan seleksi fitur Chi – Squere .')
    st.title('Klasifikasi data inputan berupa : ')
    st.write('1. Ulasan : data komentar atau ulasan yang diambil dari Google Maps')
    st.write('2. Sentimen: kelas keluaran [1: positif, 0: Negatif]')
    st.title('Dataset yang akan digunakan adalah 3 wisata di Pulau Madura yaitu : ')
    st.write('1. Pantai Sembilan dari Sumenep')
    st.write('2. Air Terjun Toroan dari Sampang')
    st.write('3. Bukit Jaddhih dari Bangkalan')
    st.write('4. Data gabungan 3 wisata')

with Learn:
    st.title('Data Wisata Pantai Sembilan: ')
    st.write('Total jumlah ulasan : 932 ')
    st.write('Dengan sentimen positif sebanyak 866 ulasan dan sentimen negatif sebanyak 66 ulasan')
    data1=pd.read_excel('dataset/Pantai_Sembilan.xlsx')
    label=[]
    for index, row in data1.iterrows():
        if row["sentimen"] == 'Positif':
            label.append(1)
        else:
            label.append(0)

    data1['label']=label
    data1=data1.drop(columns=['sentimen'])
    data1

    st.title('Data Wisata Air Terjun Toroan: ')
    st.write('Total jumlah ulasan : 877 ')
    st.write('Dengan sentimen positif sebanyak 694 ulasan dan sentimen negatif sebanyak 183 ulasan')
    data2=pd.read_excel('dataset/air_terjun_teroan.xlsx')
    label=[]
    for index, row in data2.iterrows():
        if row["sentimen"] == 'Positif':
            label.append(1)
        else:
            label.append(0)

    data2['label']=label
    data2=data2.drop(columns=['sentimen'])
    data2

    st.title('Data Wisata Bukit Jaddih: ')
    st.write('Total jumlah ulasan : 978 ')
    st.write('Dengan sentimen positif sebanyak 780 ulasan dan sentimen negatif sebanyak 198 ulasan')
    data3=pd.read_excel('dataset/bukit_jaddih.xlsx')
    label=[]
    for index, row in data3.iterrows():
        if row["sentimen"] == 'Positif':
            label.append(1)
        else:
            label.append(0)

    data3['label']=label
    data3=data3.drop(columns=['sentimen'])
    data3

    st.title('Data Gabungan 3 Wisata : ')
    st.write('Total jumlah ulasan : 2787 ')
    st.write('Dengan sentimen positif sebanyak 2340 ulasan dan sentimen negatif sebanyak 447 ulasan')
    data4=pd.read_excel('dataset/data_baru.xlsx')
    label=[]
    for index, row in data4.iterrows():
        if row["sentimen"] == 'Positif':
            label.append(1)
        else:
            label.append(0)

    data4['label']=label
    data4=data4.drop(columns=['sentimen'])
    data4
with Proses:
   st.title("""Preprosessing""")
   clean_tag = re.compile('@\S+')
   clean_url = re.compile('https?:\/\/.*[\r\n]*')
   clean_hastag = re.compile('#\S+')
   clean_symbol = re.compile('[^a-zA-Z]')
   def clean_punct(text):
      #menghapus tab,new line dan back slice
    text=text.replace('\\t'," ").replace('\\n', " ").replace('\\u'," ").replace('\\'," ")
    #menghapus no ASCII(emoticon, chines word, etc)
    text=text.encode('ascii','replace').decode('ascii')
    #menghapus link, hastag
    text=' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ",text).split())
    return text.replace("http://"," ").replace("https://"," ")
   # Buat kolom tambahan untuk data description yang telah diremovepunctuation   
   preprocessing1= data1['ulasan'].apply(clean_punct)
   preprocessing2= data2['ulasan'].apply(clean_punct)
   preprocessing3= data3['ulasan'].apply(clean_punct)
   preprocessing4= data4['ulasan'].apply(clean_punct)
   clean1=pd.DataFrame(preprocessing1)
   clean2=pd.DataFrame(preprocessing2)
   clean3=pd.DataFrame(preprocessing3)
   clean4=pd.DataFrame(preprocessing4)
   "### Proses Cleaning "
   st.write("Cleaning adalah proses pembersihan dari tab, newline,back slice.")
   st.write("Hasil cleaning data Pantai Sembilan")
   clean1
   st.write("Hasil cleaning data Air Terjun Toroan")
   clean2
   st.write("Hasil cleaning data Bukit Jaddih")
   clean3
   st.write("Hasil cleaning data Gabungan 3 data")
   clean4

   def clean_lower(lwr):
      lwr = lwr.lower() # lowercase text
      return lwr
   # Buat kolom tambahan untuk data description yang telah dicasefolding  
   clean1 = clean1['ulasan'].apply(clean_lower)
   clean2 = clean2['ulasan'].apply(clean_lower)
   clean3 = clean3['ulasan'].apply(clean_lower)
   clean4 = clean4['ulasan'].apply(clean_lower)
   casefolding1=pd.DataFrame(clean1)
   casefolding2=pd.DataFrame(clean2)
   casefolding3=pd.DataFrame(clean3)
   casefolding4=pd.DataFrame(clean4)
   "### Proses Case folding "
   st.write("Proses mengubah semua huruf ke dalam huruf kecil (lower text)")
   st.write("Hasil case folding data Pantai Sembilan")
   casefolding1
   st.write("Hasil case folding Air Terjun Toroan")
   casefolding2
   st.write("Hasil case folding data Bukit Jddih")
   casefolding3
   st.write("Hasil case folding data Gabungan 3 data")
   casefolding4

   def remove_number(text):
    return re.sub(r"\d","", text)
   rn1=casefolding1['ulasan'].apply(remove_number)
   rn2=casefolding2['ulasan'].apply(remove_number)
   rn3=casefolding3['ulasan'].apply(remove_number)
   rn4=casefolding4['ulasan'].apply(remove_number)
   remove_number1=pd.DataFrame(rn1)
   remove_number2=pd.DataFrame(rn2)
   remove_number3=pd.DataFrame(rn3)
   remove_number4=pd.DataFrame(rn4)
   "### Proses Remove Number"
   st.write("Remove number adalah tahapan preprocessing text yang bertujuan untuk membersikan teks dari angka")
   st.write("Hasil remove number data Pantai Sembilan")
   remove_number1
   st.write("Hasil remove number Air Terjun Toroan")
   remove_number2
   st.write("Hasil remove number data Bukit Jddih")
   remove_number3
   st.write("Hasil remove number data Gabungan 3 data")
   remove_number4


   def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))
   rp1=remove_number1['ulasan'].apply(remove_punctuation)
   rp2=remove_number2['ulasan'].apply(remove_punctuation)
   rp3=remove_number3['ulasan'].apply(remove_punctuation)
   rp4=remove_number4['ulasan'].apply(remove_punctuation)
   remove_punctuation1=pd.DataFrame(rp1)
   remove_punctuation2=pd.DataFrame(rp2)
   remove_punctuation3=pd.DataFrame(rp3)
   remove_punctuation4=pd.DataFrame(rp4)
   "### Proses Remove punctuation"
   st.write("Remove punctuation adalah tahapan preprocessing text yang bertujuan untuk menghapus tanda baca")
   st.write("Hasil remove punctuation data Pantai Sembilan")
   remove_punctuation1
   st.write("Hasil remove punctuation Air Terjun Toroan")
   remove_punctuation2
   st.write("Hasil remove punctuation data Bukit Jddih")
   remove_punctuation3
   st.write("Hasil remove punctuation data Gabungan 3 data")
   remove_punctuation4

   def word_tokenize_warpper(text):
    return word_tokenize(text)
   token1=remove_punctuation1['ulasan'].apply(word_tokenize_warpper)
   token2=remove_punctuation2['ulasan'].apply(word_tokenize_warpper)
   token3=remove_punctuation3['ulasan'].apply(word_tokenize_warpper)
   token4=remove_punctuation4['ulasan'].apply(word_tokenize_warpper)
   word_tokenize_warpper1=pd.DataFrame(token1)
   word_tokenize_warpper2=pd.DataFrame(token2)
   word_tokenize_warpper3=pd.DataFrame(token3)
   word_tokenize_warpper4=pd.DataFrame(token4)
   "### Proses Tokenisasi"
   st.write("Tokenisasi adalah Tahap Pemotongan teks menjadi kata")
   st.write("Hasil Tokenisasi data Pantai Sembilan")
   word_tokenize_warpper1
   st.write("Hasil Tokenisasi Air Terjun Toroan")
   word_tokenize_warpper2
   st.write("Hasil Tokenisasi data Bukit Jaddih")
   word_tokenize_warpper3
   st.write("Hasil Tokenisasi data Gabungan 3 data")
   word_tokenize_warpper4

   normalizad_word=pd.read_csv("https://raw.githubusercontent.com/135-ShintaNuriyatulMahmudiyah/cobaa/main/colloquial-indonesian-lexicon.csv")
   normalizad_word_dict={}
   for index, row in normalizad_word.iterrows():
    if row [0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]]=row[1]
   def normalized_term(document):
        return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]
   normalisasi1=word_tokenize_warpper1['ulasan'].apply(normalized_term)
   normalisasi2=word_tokenize_warpper2['ulasan'].apply(normalized_term)
   normalisasi3=word_tokenize_warpper3['ulasan'].apply(normalized_term)
   normalisasi4=word_tokenize_warpper4['ulasan'].apply(normalized_term)
   normalization1=pd.DataFrame(normalisasi1)
   normalization2=pd.DataFrame(normalisasi2)
   normalization3=pd.DataFrame(normalisasi3)
   normalization4=pd.DataFrame(normalisasi4)
   "### Proses Normalisasi"
   st.write("Normalisasi adalah Proses perbaikan kata yang typo karena disingkat")
   st.write("Hasil Normalisasi data Pantai Sembilan")
   normalization1
   st.write("Hasil Normalisasi Air Terjun Toroan")
   normalization2
   st.write("Hasil Normalisasi data Bukit Jaddih")
   normalization3
   st.write("Hasil Normalisasi data Gabungan 3 data")
   normalization4

   def remove_char(text):
    return " ".join ([w for w in text if len(w)>3])
   filtering1=normalization1['ulasan'].apply(remove_char)
   filtering2=normalization2['ulasan'].apply(remove_char)
   filtering3=normalization3['ulasan'].apply(remove_char)
   filtering4=normalization4['ulasan'].apply(remove_char)
   filtering1=pd.DataFrame(filtering1)
   filtering1=filtering1['ulasan'].apply(word_tokenize_warpper)
   filtering2=pd.DataFrame(filtering2)
   filtering2=filtering2['ulasan'].apply(word_tokenize_warpper)
   filtering3=pd.DataFrame(filtering3)
   filtering3=filtering3['ulasan'].apply(word_tokenize_warpper)
   filtering4=pd.DataFrame(filtering4)
   filtering4=filtering4['ulasan'].apply(word_tokenize_warpper)
   "### Proses Filtering"
   st.write("Filtering adalah Proses  dari empat huruf")
   st.write("Hasil Filtering data Pantai Sembilan")
   filtering1
   st.write("Hasil Filtering Air Terjun Toroan")
   filtering2
   st.write("Hasil Filtering data Bukit Jaddih")
   filtering3
   st.write("Hasil Filtering data Gabungan 3 data")
   filtering4

   def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    stemmed_text = []
    for sentence in text:
        stemmed_words = [stemmer.stem(word) for word in sentence]
        stemmed_text.append(stemmed_words)

    return stemmed_text
   "### Proses stemming"
   st.write("stemming adalah Penyeleksian kata berimbuhan menjadi kata dasar")
   stemming1 = stemming(filtering1)
   stemming2 = stemming(filtering2)
   stemming3 = stemming(filtering3)
   stemming4 = stemming(filtering4)
   stemming1
   stemming2
   stemming3
   stemming4

   def stopword(text):
    stopword=[]
    for i in range(len(text)):
        listStopword =  set(stopwords.words('indonesian')+stopwords.words('english'))
        removed=[]
        for x in (text[i]):
            if x not in listStopword:
                removed.append(x)
        stopword.append(removed)
    return removed
   "### Proses Stopword"
   st.write("Stopword adalah kata umum yang biasanya muncul dalam jumlah besar dan dianggap tidak memiliki makna")
   stopword1 = stopword(stemming1)
   stopword2 = stopword(stemming2)
   stopword3 = stopword(stemming3)
   stopword4 = stopword(stemming4)
   stopword1
   stopword2
   stopword3
   stopword4
   
with Model:
   st.title("""TF-IDF dan Model""")
   # Setiap kata dalam teks dijadikan fitur
   X=[','.join(map(str,l)) for l in stemming1['ulasan']]#X menjadi variabel fitur
   X=np.array(X)# mengubah X menjadi array
   y=df['label'].to_list()#mengubah label menjadi list dan y ini menjadi target

   # Melakukan observasi pada variabel fitur yang hasilnya akan dijadikan array
   vect=CountVectorizer()
   X_dtm=vect.fit_transform(X)
   X_dtm=X_dtm.toarray()

   # Membuat dataframe berdasarkan hasil observasi pada setiap fitur
   word_list = pd.DataFrame(X_dtm, columns=vect.get_feature_names_out())
   word_list

   # Menghitung nilai rata-rata TF-IDF untuk setiap kata
   tfidf_means = word_list.mean()

   # Mengambil top 10 kata dengan nilai TF-IDF tertinggi
   top_10_tfidf = tfidf_means.nlargest(10)

   # Membuat grafik untuk menampilkan top 10 TF-IDF
   st.title("Grafik untuk menampilkan top 10 TF-IDF Pantai Sembilan")
   plt.figure(figsize=(10, 6))
   top_10_tfidf.plot(kind='bar', color='skyblue')
   plt.title('Top 10 TF-IDF Values')
   plt.xlabel('Words')
   plt.ylabel('TF-IDF Value')
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.show()
   st.title("Hasil klasifikasi Logistic Regression tanpa seleksi fitur")
   X = word_list # Ganti ini dengan fitur yang sesuai
   y = df['label']  # Ganti ini dengan label yang sesuai
   # Pisahkan data menjadi data latih dan data uji
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Inisialisasi dan latih model Logistic Regression
   LR = LogisticRegression()
   LR.fit(X_train, y_train)

   # Melakukan prediksi pada data uji
   predictions = LR.predict(X_test)
   # Mengukur akurasi model sebelum seleksi fitur
   accuracy = accuracy_score(y_test, predictions)
   st.write("Akurasi Model Sebelum Seleksi Fitur: {:.2f}%".format(accuracy * 100))

   # Menampilkan laporan klasifikasi
   st.write("Classification Report:")
   st.write(classification_report(y_test, predictions))

   # Membuat confusion matrix
   confusion = confusion_matrix(y_test, predictions)
   class_label=["Positif","Negatif"]
   test = pd.DataFrame(confusion, index = class_label, columns =
   class_label)
   sns.heatmap(test, annot = True,fmt="d")
   plt.title("Confusion Matrix Logistic Regression Without Feature Selection")
   plt.xlabel("Predicted Label")
   plt.ylabel("True Label")
   plt.show()

   st.write("Confusion Matrix:")
   st.write(confusion)

   st.title("Hasil klasifikasi Logistic Regression dengan seleksi fitur")
   # Fungsi untuk melakukan seleksi fitur menggunakan Chi-squere
   def select_features(matrix, labels, k=500):
    selector = SelectKBest(chi2, k=k)
    selected_features = selector.fit_transform(matrix, labels)
    return selector.get_support(indices=True)
   
   X = X_dtm # Ganti ini dengan fitur yang sesuai
   y = df['label']   # Ganti ini dengan label yang sesuai

   # Seleksi fitur menggunakan Chi-Squere
   selected_features=select_features(X,y)
   # Pisahkan data menjadi data latih dan data uji
   X_train, X_test, y_train, y_test = train_test_split(X[:, selected_features], y, test_size=0.2, random_state=42)

   # Inisialisasi dan latih model Logistic Regression
   LR = LogisticRegression()
   LR.fit(X_train, y_train)

   # Melakukan prediksi pada data uji
   predictions = LR.predict(X_test)


   # Mengukur akurasi model setelah seleksi fitur
   accuracy = accuracy_score(y_test, predictions)
   st.write("Akurasi Model Setelah Seleksi Fitur: {:.2f}%".format(accuracy * 100))

   # Menampilkan laporan klasifikasi
   st.write("Classification Report:")
   st.write(classification_report(y_test, predictions))
   # Membuat confusion matrix
   confusion = confusion_matrix(y_test, predictions)
   class_label=["Positif","Negatif"]
   test = pd.DataFrame(confusion, index = class_label, columns =class_label)
   sns.heatmap(test, annot = True,fmt="d")
   plt.title("Confusion Matrix Logistic Regression WithFeature Selection")
   plt.xlabel("Predicted Label")
   plt.ylabel("True Label")
   plt.show()

   st.write("Confusion Matrix:")
   st.write(confusion)


   # Setiap kata dalam teks dijadikan fitur
   X=[','.join(map(str,l)) for l in stemming2['ulasan']]#X menjadi variabel fitur
   X=np.array(X)# mengubah X menjadi array
   y=df['label'].to_list()#mengubah label menjadi list dan y ini menjadi target

   # Melakukan observasi pada variabel fitur yang hasilnya akan dijadikan array
   vect=CountVectorizer()
   X_dtm=vect.fit_transform(X)
   X_dtm=X_dtm.toarray()

   # Membuat dataframe berdasarkan hasil observasi pada setiap fitur
   word_list = pd.DataFrame(X_dtm, columns=vect.get_feature_names_out())
   word_list

   # Menghitung nilai rata-rata TF-IDF untuk setiap kata
   tfidf_means = word_list.mean()

   # Mengambil top 10 kata dengan nilai TF-IDF tertinggi
   top_10_tfidf = tfidf_means.nlargest(10)

   # Membuat grafik untuk menampilkan top 10 TF-IDF
   st.title("Grafik untuk menampilkan top 10 TF-IDF air terjun toroan")
   plt.figure(figsize=(10, 6))
   top_10_tfidf.plot(kind='bar', color='skyblue')
   plt.title('Top 10 TF-IDF Values')
   plt.xlabel('Words')
   plt.ylabel('TF-IDF Value')
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.show()


   # Setiap kata dalam teks dijadikan fitur
   X=[','.join(map(str,l)) for l in stemming3['ulasan']]#X menjadi variabel fitur
   X=np.array(X)# mengubah X menjadi array
   y=df['label'].to_list()#mengubah label menjadi list dan y ini menjadi target

   # Melakukan observasi pada variabel fitur yang hasilnya akan dijadikan array
   vect=CountVectorizer()
   X_dtm=vect.fit_transform(X)
   X_dtm=X_dtm.toarray()

   # Membuat dataframe berdasarkan hasil observasi pada setiap fitur
   word_list = pd.DataFrame(X_dtm, columns=vect.get_feature_names_out())
   word_list

   # Menghitung nilai rata-rata TF-IDF untuk setiap kata
   tfidf_means = word_list.mean()

   # Mengambil top 10 kata dengan nilai TF-IDF tertinggi
   top_10_tfidf = tfidf_means.nlargest(10)

   # Membuat grafik untuk menampilkan top 10 TF-IDF
   st.title("Grafik untuk menampilkan top 10 TF-IDF bukit jaddih")
   plt.figure(figsize=(10, 6))
   top_10_tfidf.plot(kind='bar', color='skyblue')
   plt.title('Top 10 TF-IDF Values')
   plt.xlabel('Words')
   plt.ylabel('TF-IDF Value')
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.show()


   # Setiap kata dalam teks dijadikan fitur
   X=[','.join(map(str,l)) for l in stemming4['ulasan']]#X menjadi variabel fitur
   X=np.array(X)# mengubah X menjadi array
   y=df['label'].to_list()#mengubah label menjadi list dan y ini menjadi target

   # Melakukan observasi pada variabel fitur yang hasilnya akan dijadikan array
   vect=CountVectorizer()
   X_dtm=vect.fit_transform(X)
   X_dtm=X_dtm.toarray()

   # Membuat dataframe berdasarkan hasil observasi pada setiap fitur
   word_list = pd.DataFrame(X_dtm, columns=vect.get_feature_names_out())
   word_list

   # Menghitung nilai rata-rata TF-IDF untuk setiap kata
   tfidf_means = word_list.mean()

   # Mengambil top 10 kata dengan nilai TF-IDF tertinggi
   top_10_tfidf = tfidf_means.nlargest(10)

   # Membuat grafik untuk menampilkan top 10 TF-IDF
   st.title("Grafik untuk menampilkan top 10 TF-IDF data gabungan 3 wisata")
   plt.figure(figsize=(10, 6))
   top_10_tfidf.plot(kind='bar', color='skyblue')
   plt.title('Top 10 TF-IDF Values')
   plt.xlabel('Words')
   plt.ylabel('TF-IDF Value')
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.show()