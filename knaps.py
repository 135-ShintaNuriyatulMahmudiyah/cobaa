import streamlit as st
from PIL import Image
import cv2

st.set_page_config(
    page_title="Project"
)
st.title('Classification Of Rice Leaves')

tab1, tab2 = st.tabs(["Information", "Test Image"])

with tab1:
    st.subheader("Pada klasifikasi ini terbagi menjadi 4 kelas atau label yaitu :")
    st.write("""
    <ol>
        <li>Brown Spot : Penyakit bercak daun coklat pada tanaman padi yaitu Oryzae berwarna coklat, bersekat 6-17, berbentuk silindris, agak melengkung, dan bagian tengahnya agak melebar.</li>
        <li>Hispa: penyakit yang memiliki bercak putih besar akibat serangan serangga dewasa yang mengikis permukaan daun.</li>
        <li>Leaf Blast: penyakit yang memiliki bercak kuning pada bagian ujung, hingga berwarna kecoklatan dan juga kering pada tanaman.</li>
        <li>Healthy : Memiliki warna hijau cerah, bentuk yang khas, permukaan yang halus, dan struktur pembuluh daun yang terlihat jelas.</li>
    </ol>""",unsafe_allow_html=True)
with tab2:
        # Menu pilihan
        menu = st.selectbox("Capture Option :",["Upload Photo", "Camera"])

        if menu == "Upload Photo":
            uploaded_file = st.file_uploader("Select photo", type=['png', 'jpg', 'jpeg'])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Photo', use_column_width=True)
                # Lakukan pemrosesan gambar di sini (jika diperlukan)

        elif menu == "Camera":
            st.write("Click the camera button below.")
            if st.button('Camera'):
                
                # Buat objek kamera
                cap = st.camera_input("Take a picture")

                # Baca frame kamera secara berulang-ulang
                if cap is not None:
                        # To read image file buffer with OpenCV:
                        bytes_data = cap.getvalue()
                        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    
                        # Check the type of cv2_img:
                        # Should output: <class 'numpy.ndarray'>
                        st.write(type(cv2_img))
                    
                        # Check the shape of cv2_img:
                        # Should output shape: (height, width, channels)
                        st.write(cv2_img.shape)

