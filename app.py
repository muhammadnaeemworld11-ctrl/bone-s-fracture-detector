import streamlit as st
from ultralytics import YOLO
from PIL import Image

@st.cache_resource
def load_model():
    return YOLO("best.pt")

new_model = load_model()

st.title("🛡️ Weapon Detection App")
st.write("Upload an X-ray or optical image to scan for concealed weapons.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    input_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)
        
    with col2:
        st.subheader("Object Detection")
        
        if st.button("Run Threat Scan", use_container_width=True):
            with st.spinner("Scanning for weapons..."):
                results = new_model(input_image)       
            
            st.image(
                results[0].plot(),  
                use_container_width=True, 
                caption="Scan Analysis Complete"
            )
