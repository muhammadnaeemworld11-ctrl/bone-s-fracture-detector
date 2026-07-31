import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("Object Detection App")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    input_image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)
        
    with col2:
        st.subheader("Results")
        if st.button("Run Detection", use_container_width=True):
            with st.spinner("Processing..."):
                results = model(input_image)       
            
            st.image(
                results[0].plot(),  
                use_container_width=True
            )
            
            if len(results[0].boxes) > 0:
                st.info(f"Detected {len(results[0].boxes)} object(s).")
            else:
                st.write("No objects detected.")
