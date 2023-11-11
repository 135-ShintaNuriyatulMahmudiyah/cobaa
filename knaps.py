import streamlit as st
import pandas as pd
import numpy as np
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


st.title("Web Apps - Analisis Sentimen")

st.write("================================================================================")

st.write("Name :Shinta Nuriyatul Mahmudiyah")
st.write("Nim  :200411100135")
st.write("Grade: Penambangan Data A")

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
    df = pd.read_csv('https://raw.githubusercontent.com/135-ShintaNuriyatulMahmudiyah/Data/main/data_baru.csv', sep='\t')
    st.write(df.head())

with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)

    #Mendefinisikan Varible X dan Y
    X = df[["lokasi","ulasan","sentimen"]]
    y = df["fruit_name"].values
    df
    X
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.fruit_name).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1' : [dumies[0]],
        '2' : [dumies[1]],
        '3' : [dumies[2]],
        '4' : [dumies[3]],
        
    })

    st.write(labels)

   
with modeling:
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
  
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        mass = st.number_input('Masukkan berat buah (mass) : ')
        width = st.number_input('Masukkan lebar buah (width) : ')
        height = st.number_input('Masukkan tinggi buah (height) : ')
        color_score = st.number_input('Masukkan skor warna (color_score) : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                mass,
                width,
                height,
                color_score
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
