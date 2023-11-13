import streamlit as st
import pandas as pd
import numpy as np
import string
#import nltk
import re
#from wordcloud import WordCloud
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from nltk.corpus import stopwords
#import matplotlib.pyplot as plt
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
    df = pd.read_csv('https://raw.githubusercontent.com/135-ShintaNuriyatulMahmudiyah/Data/main/Data.csv',sep='\t')
    if df is not None:
        df= Analyzer(df)
        st.subheader('Pre-Processing text data')
        with st.spinner('With text processing in Progress....'):
            with st.expander("Expand for details"):
                st.subheader('Drop duplicate raws & lower casing all text')
                df=df.drop_duplicates(inplace=True)
                #mengubah semua huruf ke dalam huruf kecil (lower text)
                df['clean_ulasan']=df['ulasan'].str.lower()
                st.table(df[['clean_ulasan','ulasan']].head(5))
    
                                









                
