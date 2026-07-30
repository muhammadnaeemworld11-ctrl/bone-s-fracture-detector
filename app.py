import PIL.Image
import streamlit as st
from ultralytics import YOLO

new_model = YOLO("best.pt")

st.title("Bone's Fracture Detection App")
st.write("Upload an image to detect bone's fractures")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    input_image = PIL.Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)

    with col2:
        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                results = new_model(input_image)
                res_plotted = results[0].plot()

                image = st.image(
                    res_plotted, channels="BGR", use_container_width=True, caption="Predicted Image"
                )
                
# docker build -t bone-fracture-app .
# docker run -p 8501:8501 bone-fracture-app
