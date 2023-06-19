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

st.set_page_config(
    page_title="Project"
)
st.title('Web-Apps Classification Of Rice Leaves')

st.write("================================================================================")



data_set_description, data, preprocessing, modeling, implementation = st.tabs(["Data Set Description", "Data", "Preprocessing", "Modeling", "Implementation"])

with data_set_description:
     st.subheader("Pada klasifikasi ini terbagi menjadi 4 kelas atau label yaitu :")
    st.write("<ol>
        <li>Brown Spot : Penyakit bercak daun coklat pada tanaman padi yaitu Oryzae berwarna coklat, bersekat 6-17, berbentuk silindris, agak melengkung, dan bagian tengahnya agak melebar.</li>
        <li>Hispa: penyakit yang memiliki bercak putih besar akibat serangan serangga dewasa yang mengikis permukaan daun.</li>
        <li>Leaf Blast: penyakit yang memiliki bercak kuning pada bagian ujung, hingga berwarna kecoklatan dan juga kering pada tanaman.</li>
        <li>Healthy : Memiliki warna hijau cerah, bentuk yang khas, permukaan yang halus, dan struktur pembuluh daun yang terlihat jelas.</li></ol>",unsafe_allow_html=True)

with data:
    df = pd.read_csv('https://raw.githubusercontent.com/elmatiaaa/prosaindata/main/new.csv')
    st.dataframe(df)

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
    

    
    df = df.drop(columns=['Date'])
    #Mendefinisikan Varible X dan Y
    X = df[['Open','High','Low','Close','AdjClose']]
    y = df["Volume"].values
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
    dumies = pd.get_dummies(df.Volume).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        '1' : [dumies[0]],
        '2' : [dumies[1]],
        '3' : [dumies[2]],
        '4' : [dumies[3]],
        '4' : [dumies[4]],
        
    })

    st.write(labels)

   
with modeling:
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing

    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

       
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
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [knn_akurasi, dt_akurasi],
                'Model' : [ 'K-NN', 'Decission Tree'],
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
        Open = st.number_input('input Open : ')
        High = st.number_input('Input High : ')
        Low= st.number_input('Input Low : ')
        Close = st.number_input('Input Close : ')
        AdjClose = st.number_input('Input AdjClose : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ( 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                'Open',
                'High',
                'Low',
                'Close',
                'AdjClose'
               
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)
\
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
