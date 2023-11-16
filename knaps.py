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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
#import seaborn as sns

# import warnings
# warnings.filterwarnings("ignore")


st.title("Sentiment Analysis - Web Apps")
st.write("""
#### Analisis Sentimen Ulasan Wisata di Pulau Madura Menggunakan Logistic Regression dan Seleksi fitur Chi-Squere
Berapa Nilai Akurasi yang dihasilkan?
""")

st.write("================================================================================")

#st.write("Name :Shinta Nuriyatul Mahmudiyah")
#st.write("Nim  :200411100135")
#st.write("Grade: Penambangan Data A")
st.write("""### Data Set Description """)


data_set_description, data, preprocessing, modeling, implementation = st.tabs(["Data Set Description", "Data", "Preprocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("""# Data Set Description """)
    st.write("###### Data Set Ini Adalah : Fruit with Color ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/mjamilmoughal/fruits-with-colors-dataset")
    st.write("""Dalam dataset ini terdapat 59 data dan 7 kolom yaitu fruit label, fruit name, fruit subtype, mass width, height, dan color score. Untuk
     dataset ini mempunyai 4 kelas yaitu Apple, Mandarin, Orange, dan Lemon. 
   
    """)
    st.write("""###### Penjelasan setiap kolom : """)
    st.write("""1. Fruit Label (Label Buah) :
    Label Buah ini merupakan pengganti nama buah. Berikur penjelasan:
    1. Apel
    2. Mandarin
    3. Orange
    4. Lemon
   
    """)
    st.write("""2. Fruit Name (Nama Buah) :
    ini akan menjadi outputnya yaitu nama buah.Dalam Aplikasi ini akan nama buah yang akan diprediksi ada 4 yaitu Apple, Orange, Mandarin, dan Lemon.
   
    """)
    
    st.write("""3. Fruit Subtype (Tipe Buah) :
    Ini merupakan tipe buah. untuk buah apel, mandarin, orange, dan lemon mempunyai tipe buah yang berbeda- beda. 
   
    """)
    st.write("""4. Mass (Massa Buah) :
    setiap buah mempunyai berat dengan satuan gram. setiap buah juga mempunyai massa buah yang berbeda - beda.
    
    """)
    st.write("""5. Width (Lebar Buah):
    setiap buah mempunyai lebar buah yang berbeda - beda.
    
    """)
    st.write("""6. Height (Tinggi Buah):
    setiap buah mempunyai tinggi buah yang berbeda - beda.
    
    """)
    st.write("""7. Color_Score (Skor Warna) :
    setiap buah mempunyai skor warna  yang berbeda - beda.
    
    """)
    st.write("""Dari inputan Massa, Width, Height, dan Color_Score itu akan menghasilkan output nama buah
    
    """)
    
    st.write("""Memprediksi Nama Buah (output) :

    1. Apple 
    2. Mandarin 
    3. Orange 
    4. Lemon 
    """)
    st.write("###### Aplikasi ini untuk : Fruit  Prediction (Prediksi buah) ")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : https://github.com/135-ShintaNuriyatulMahmudiyah/PenambanganDataWeb ")
    st.write("###### Untuk Wa saya anda bisa hubungi nomer ini : http://wa.me/6285704097096 ")

with data:
    df = pd.read_csv('https://raw.githubusercontent.com/135-ShintaNuriyatulMahmudiyah/Data/main/Data.csv',sep='\t')
    st.dataframe(df)
with preprocessing:
     st.title("""Preprosessing""")
   clean_tag = re.compile('@\S+')
   clean_url = re.compile('https?:\/\/.*[\r\n]*')
   clean_hastag = re.compile('#\S+')
   clean_symbol = re.compile('[^a-zA-Z]')
   def clean_punct(text):
      text = clean_tag.sub('', text)
      text = clean_url.sub('', text)
      text = clean_hastag.sub(' ', text)
      text = clean_symbol.sub(' ', text)
      return text
   # Buat kolom tambahan untuk data description yang telah diremovepunctuation   
   preprocessing = data['Text'].apply(clean_punct)
   clean=pd.DataFrame(preprocessing)
   "### Melakukan Cleaning "
   clean

   def clean_lower(lwr):
      lwr = lwr.lower() # lowercase text
      return lwr
   # Buat kolom tambahan untuk data description yang telah dicasefolding  
   clean = clean['Text'].apply(clean_lower)
   casefolding=pd.DataFrame(clean)
   "### Melakukan Casefolding "
   casefolding

   def to_list(text):
      t_list=[]
      for i in range(len(text)):
         t_list.append(text[i])
      return t_list

   casefolding1 = to_list(clean)

   "### Melakukan Tokenisasi "
   def tokenisasi(text):
      tokenize=[]
      for i in range(len(text)):
         token=word_tokenize(text[i])
         tokendata = []
         for x in token :
            tokendata.append(x)
         tokenize.append(tokendata)
      return tokendata

   tokenisasi = tokenisasi(casefolding1)
   tokenisasi

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
   stopword = stopword(tokenisasi)
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
   kkk = pd.read_csv("hasil_stemming.csv")
   kkk

   
   "### Hasil Proses Pre-Prosessing "
   def gabung(test):
      join=[]
      for i in range(len(stemming)):
         joinkata = ' '.join(stemming[i])
         join.append(joinkata)
      hasilpreproses = pd.DataFrame(join, columns=['Text'])
      hasilpreproses.to_csv('hasilpreproses.csv')
      return hasilpreproses

   hasilpreproses = pd.read_csv("hasilpreproses.csv")
   hasilpreproses

   st.title("""TF-IDF""")
   tr_idf_model  = TfidfVectorizer()
   tf_idf_vector = tr_idf_model.fit_transform(hasilpreproses['Text'])
   tf_idf_array = tf_idf_vector.toarray()
   words_set = tr_idf_model.get_feature_names_out()
   df_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set)
   df_tf_idf


    """#### 1. Remove Regex (Cleansing)"""
    
    # Menghilangkan kalimat Encode
    data['ulasan'].replace(to_replace = r'\\x[0-9a-fA-F][0-9a-fA-F]', value = '', regex = True, inplace = True)
    data
    def hello (ulasan):
        print("hello world")
    
    def remove(ulasan):
        # remove stock market tickers like $GE
        ulasan = re.sub(r'\$\w*', '',str(ulasan ))
        # Remove RT/b/ yang tersisa
        ulasan = re.sub(r'\bRT\b', '', ulasan)
        ulasan  = re.sub('b\'', '', ulasan)    
        # Replace 2+ dots with space
        ulasan = re.sub(r'\.{2,}', ' ', ulasan)
        #remove @username
        ulasan = re.sub('@[^\s]+','',ulasan)
         # remove old style retweet text "RT"
        ulasan = re.sub(r'^RT[\s]+', '', ulasan)
        #remove angka
        ulasan = re.sub('[0-9]+', '', ulasan)
        #remove url
        ulasan = re.sub(r"http\S+", "", ulasan)
        # remove hashtags
        ulasan = re.sub(r'#\w*', '', ulasan)
        # Strip space, " and ' from tweet
        ulasan = ulasan.strip(' "\'')
        # Replace multiple spaces with a single space
        ulasan = re.sub(r'\s+', ' ', ulasan)
        #hapus tanda baca
        ulasan = ulasan.translate(str.maketrans("","",string.punctuation))
        #hapus karakter
        ulasan = re.sub(r'\n', '', ulasan)
    
        return ulasan 
    data['clean'] = data['ulasan'].apply(lambda x: remove(x))
     
data

"""#### 2. Case Folding"""

# proses case folding 
data['case_folding'] = data['clean'].str.lower()
data






                
