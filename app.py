import streamlit as st
from ultralytics import YOLO
from PIL import Image

new_model = YOLO("best.pt")

st.title("Bone Fracture Detection App")
st.write("Upload an image to detect bone fractures.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    input_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)
        
    with col2:
        st.subheader("Predicted Image")
        if st.button("Predict Fracture", use_container_width=True):
            with st.spinner("Analyzing..."):
                results = new_model(input_image)       
            
            st.image(
                results[0].plot(),  
                channels="BGR", 
                use_container_width=True, 
                caption="Predicted Image"
            )
