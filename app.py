import streamlit as st
from ultralytics import YOLO
from PIL import Image

new_model = YOLO("best.pt")

st.title("Bone Fracture Detection App")
st.write("Upload an image to detect bone fractures.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    input_image = Image.open(uploaded_file)
    
    if "predicted_img" not in st.session_state:
        st.session_state.predicted_img = None

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)
        s
        if st.button("Predict Fracture", use_container_width=True):
            with st.spinner("Analyzing..."):
                results = new_model(input_image)
                st.session_state.predicted_img = results[0].plot()

    with col2:
        st.subheader("Prediction Result")
        if st.session_state.predicted_img is not None:
            st.image(
                st.session_state.predicted_img, 
                channels="BGR", 
                use_container_width=True, 
                caption="Predicted Image"
            )
        else:
            st.info("Click 'Predict Fracture' to see results.")
