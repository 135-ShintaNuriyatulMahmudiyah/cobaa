import streamlit as st
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import re


Home, Learn, Proses, Model, Implementasi = st.tabs(['Home', 'Learn Data', 'Preprocessing dan TF-IDF', 'Model', 'Implementasi'])

with Home:
   st.title("""SENTIMEN ANALISIS RESESI 2023""")
   st.subheader('Kelompok 2')
   st.text("""
            1. Nuskhatul Haqqi 200411100034
            2. Zuni Amanda Dewi 200411100051
            3. Abd. Hanif Azhari 200411100101""")

with Learn:
   st.title("""Sentiment Analisis Resesi 2023""")
   st.write('Resesi dunia adalah kondisi ketika perekonomian sebagian besar negara sedang memburuk seiring menurunnya aktivitas di sektor perdagangan dan industri. Beberapa waktu lalu, menteri keuangan Sri Mulyani memproyeksikan dunia akan memasuki resesi pada tahun 2023.')
   st.write('Dalam Klasifikasi ini data yang digunakan adalah ulasan atau komentar dari aplikasi Twitter dengan topik Resesi 2023.')
   st.title('Klasifikasi data inputan berupa : ')
   st.write('1. text : data komentar atau ulasan yang diambil dari twitter')
   st.write('2. Label: kelas keluaran [1: positif, -1: Negatif]')

   st.title("""Asal Data""")
   st.write("Dataset yang digunakan adalah data hasil crowling twitter dengan kata kunci 'Resesi Ekonomi 2023' yang disimpan di https://raw.githubusercontent.com/nuskhatulhaqqi/data_mining/main/resesi_2023%20(1).csv")
   st.write("Total datanya adalah 132 dengan atribut 2")
   # uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
   # if uploaded_files is not None :
   data = pd.read_csv('https://raw.githubusercontent.com/135-ShintaNuriyatulMahmudiyah/Data/main/Data.csv')
   # else:
   #    for uploaded_file in uploaded_files:
   #       data = pd.read_csv(uploaded_file)
   #       st.write("Nama File Anda = ", uploaded_file.name)
   #       st.dataframe(data)
      


with Proses:
   st.title("""Preprosessing""")
   def remove_ulasan_Special(text):
      #menghapus tab,new line dan back slice
      text=text.replace('\\t'," ").replace('\\n', " ").replace('\\u'," ").replace('\\'," ")
      #menghapus no ASCII(emoticon, chines word, etc)
      text=text.encode('ascii','replace').decode('ascii')
      #menghapus link, hastag
      text=' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ",text).split())
      return text.replace("http://"," ").replace("https://"," ")
   remove=df['ulasan'].apply(remove_ulasan_Special)
   clean=pd.DataFrame(remove)
   "### Melakukan remove_ulasan_Special"
   clean

   def clean_lower(lwr):
      lwr = lwr.lower() # lowercase text
      return lwr
   # Buat kolom tambahan untuk data description yang telah dicasefolding  
   clean = clean['ulasan'].apply(clean_lower)
   casefolding=pd.DataFrame(clean)
   "### Melakukan Casefolding "
   casefolding

   #menghapus angka
   def remove_number(text):
     return re.sub(r"\d","", text)
   rnumber=df['ulasan'].apply(remove_number)
   remove_number=pd.DataFrame(rnumber)
   "### menghapus angka"
   remove_number

   #menghapus tanda baca
   def remove_punctuation(text):
     return text.translate(str.maketrans("","",string.punctuation))
   tanda_baca=df['ulasan'].apply(remove_punctuation)
   remove_punctuation=pd.DataFrame(tanda_baca)
   "### menghapus tanda baca"
   remove_punctuation

   

   def to_list(text):
      t_list=[]
      for i in range(len(text)):
         t_list.append(text[i])
      return t_list

   casefolding1 = to_list(clean)

   "### Melakukan Tokenisasi "
   #tokenizing
   def word_tokenize_warpper(text):
     return word_tokenize(text)
   token=df['ulasan'].apply(word_tokenize_warpper)
   tokenisasi=pd.DataFrame(token)
   "### tokenisasi"
   tokenisasi

   "### Melakukan Normalisasi "
   #normalisasi
   #menyeragamkan kata yang memiliki makna yang sama namun penelitiannya berbeda
   normalizad_word=pd.read_csv("https://raw.githubusercontent.com/135-ShintaNuriyatulMahmudiyah/cobaa/main/colloquial-indonesian-lexicon.csv")
   normalizad_word_dict={}
   for index, row in normalizad_word.iterrows():
     if row [0] not in normalizad_word_dict:
       normalizad_word_dict[row[0]]=row[1]
   def normalized_term(document):
     return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]
   normal=df['ulasan'].apply(normalized_term)
   normalisasi=pd.DataFrame(normal)
   "### Normalisasi"
   normalisasi


   "### Melakukan Stopword Removal "
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
   stopword = stopword(token)
   stopword
   "### Melakukan Stemming "
   def stemming(text):
      stemming=[]
      for i in range(len(text)):
         factory = StemmerFactory()
         stemmer = factory.create_stemmer()
         katastem=[]
         for x in (text[i]):
            katastem.append(stemmer.stem(x))
         stemming.append(katastem)
      return stemming
   # kk = pd.DataFrame(stemming)
   # kk.to_csv('hasil_stemming.csv')
   #kkk = pd.read_csv("hasil_stemming.csv")
   #kkk

   
   "### Hasil Proses Pre-Prosessing "
   data = pd.read_csv('https://raw.githubusercontent.com/135-ShintaNuriyatulMahmudiyah/cobaa/main/hasilpreproses.csv')
   import ast
   def join(texts):
     return " ".join([hasilpreproses for hasilpreproses in texts])
   df['hasilpreproses']=df['hasilpreproses'].apply(join)
   df.head()
   df['hasilpreproses'].to_csv('hasilpreproses.csv')

   #hasilpreproses = pd.read_csv("hasilpreproses.csv")
   #hasilpreproses

   st.title("""TF-IDF""")
