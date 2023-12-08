import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn import svm
from nltk.corpus import stopwords
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


Home, Learn, Proses, Model, Implementasi = st.tabs(['Home', 'Learn Data', 'Preprocessing dan TF-IDF', 'Seleksi Fitur','Model', 'Implementasi'])

with Home:
   st.title("""ANALISIS SENTIMEN  ULASAN PENGUNJUNG WISATA DI PULAU MADURA MENGGUNAKAN METODE LOGISTIC REGREESION DENGAN SELEKSI FITUR CHI – SQUERE""")
   st.subheader('Oleh :')
   st.text("""Shinta Nuriyatul Mahmudiyah - 200411100135""")

with Learn:
   st.title("""ANALISIS SENTIMEN  ULASAN PENGUNJUNG WISATA DI PULAU MADURA MENGGUNAKAN METODE LOGISTIC REGREESION DENGAN SELEKSI FITUR CHI – SQUERE""")
   st.write('Parwisata merupakan sektor penting dalam mengembangkan perekonomian suatu daerah. Pulau madura ini mempunyai keindahan alam yang sangat indah. Namun, masih banyak wisata yang perlu diperhatikan dan diperbaiki agar banyak wisatawan mengunjungi wisata tersebut')
   st.write('Dalam Klasifikasi ini data yang digunakan adalah ulasan atau komentar dari google maps dengan bantuan extension chrome yaitu instant data scrapper.')
   st.title('Klasifikasi data inputan berupa : ')
   st.write('1. text : data komentar atau ulasan yang diambil dari google maps')
   st.write('2. Label: kelas keluaran [1: positif, -1: Negatif]')

   st.title("""Asal Data""")
   st.write("Dataset yang digunakan adalah data wisata Pantai Sembilan, Air Terjun Toroan, Bukit Jaddih, dan data gabungan dari 3 wisata tersebut")
   st.write("Total data wisata pantai sembilan adalah 932")
   st.write("Total data wisata air terjun toroan adalah 877")
   st.write("Total data wisata bukit jaddhih adalah 978")
   st.write("Total data gabungan 3 wisat atersebut adalah 2787")
   # uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
   # if uploaded_files is not None :
   data = pd.read_exel('https://github.com/135-ShintaNuriyatulMahmudiyah/cobaa/blob/main/Pantai_Sembilan.xlsx')
   # else:
   #    for uploaded_file in uploaded_files:
   #       data = pd.read_csv(uploaded_file)
   #       st.write("Nama File Anda = ", uploaded_file.name)
   #       st.dataframe(data)
      

