import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# import warnings
# warnings.filterwarnings("ignore")


st.title("PENAMBANGAN DATA")
st.write("##### Nama  : Hambali Fitrianto ")
st.write("##### Nim   : 200411100074 ")
st.write("##### Kelas : Penambangan Data C ")

data_set_description, upload_data, preprocessing, modeling, implementation = st.tabs(["Data Set Description", "Upload Data", "Preprocessing", "Modeling", "Implementation"])

with data_set_description:
    st.write("""# Data Set Description """)
    st.write("###### Data Set Ini Adalah : Weather Prediction (Prediksi Cuaca) ")
    st.write("###### Sumber Data Set dari Kaggle : https://www.kaggle.com/datasets/ananthr1/weather-prediction")
    st.write("""###### Penjelasan setiap kolom : """)
    st.write("""1. preciptation (curah hujan) :

    Curah hujan : jumlah hujan yang turun pada suatu daerah dalam waktu tertentu. untuk menentukan besarnya curah hujan, membutuhkan suatu alat ukur. Alat pengukur curah hujan disebut dengan fluviograf dan satuan curah hujan yang biasanya digunakan adalah milimeter (mm).
    """)
    st.write("""2. tempmax (suhu maks) :

    Suhu Maksimum : Suhu yang terbaca dari termometer maksimum di ada di dataset
    """)
    st.write("""3. tempmin (suhu min) :

    Suhu Minimum : Suhu yang terbaca dari termometer minimum di ada di dataset
    """)
    st.write("""4. wind (angin) :

    Kecepatan angin disebabkan oleh pergerakan angin dari tekanan tinggi ke tekanan rendah, biasanya karena perubahan suhu
    """)
    st.write("""5. weather (cuaca) :

    Output (keluaran)
    """)
    st.write("""Menggunakan Kolom (input) :

    precipitation
    tempmax * tempmin
    wind
    """)
    st.write("""Memprediksi kondisi cuaca (output) :

    1. drizzle (gerimis)
    2. rain (hujan)
    3. sun (matahari)
    4. snow (salju)
    5. fog (kabut)
    """)
    st.write("###### Aplikasi ini untuk : Weather Prediction (Prediksi Cuaca) ")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : https://github.com/HambaliFitrianto/Aplikasi-Web-Data-Mining-Weather-Prediction ")
    st.write("###### Untuk Wa saya anda bisa hubungi nomer ini : http://wa.me/6282138614807 ")

with upload_data:
    # uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    # for uploaded_file in uploaded_files:
    #     df = pd.read_csv(uploaded_file)
    #     st.write("Nama File Anda = ", uploaded_file.name)
    #     st.dataframe(df)
    df = pd.read_table('https://raw.githubusercontent.com/135-ShintaNuriyatulMahmudiyah/Data/main/fruit_data_with_colors.txt')
    st.dataframe(df)

with preprocessing:
    st.subheader("""Normalisasi Data""")
    data['fruit_name'].value_counts()
    X=data.drop(columns=['fruit_name','fruit_subtype'],axis=1)

    X
    
    x = data[["mass","width","height","color_score"]]
    y = data["fruit_label"].values
    
    st.write("""# Normalisasi MinMaxScaler""")
    "### Mengubah skala nilai terkecil dan terbesar dari dataset ke skala tertentu.pada dataset ini skala terkecil = 0, skala terbesar= 1"
    
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    x_scaled= scaler.fit_transform(x)
    x_scaled

with modeling:
    x_train, x_test,y_train,y_test= train_test_split(x,y,random_state=0)    
    x_train_scaled, x_test_scaled,y_train_scaled,y_test_scaled= train_test_split(x_scaled,y,random_state=0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    # from sklearn.feature_extraction.text import CountVectorizer
    # cv = CountVectorizer()
    # x_train = cv.fit_transform(x_train)
    # x_test = cv.fit_transform(x_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")
    
    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(x_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(x_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    # prediction
    dt.score(x_test, y_test)
    y_pred = dt.predict(x_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)
  
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        Temp_Max = st.number_input('Masukkan tempmax (suhu maks) : ')
        Temp_Min = st.number_input('Masukkan tempmin (suhu min) : ')
        Wind = st.number_input('Masukkan wind (angin) : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                mass,
                widht,
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
