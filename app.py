import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Weapon Detection App",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    new_model = load_model()
except Exception as e:
    st.error(f"Error loading model weights: {e}. Please make sure 'best.pt' is in your application directory.")
    st.stop()

st.title("🛡️ Weapon Detection App")
st.write("Upload an X-ray or optical baggage scan to detect hidden threats or weapons.")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    input_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)
        
    with col2:
        st.subheader("Object Detection")
        
        if st.button("Run Threat Scan", use_container_width=True):
            with st.spinner("Analyzing image for weapons..."):
                results = new_model(input_image, conf=0.25)       
            
            annotated_image = results[0].plot()
            
            st.image(
                annotated_image,  
                use_container_width=True, 
                caption="Scan Analysis Complete"
            )
            
            if len(results[0].boxes) > 0:
                st.warning(f"⚠️ Warning: Detected {len(results[0].boxes)} potential threat(s) in the scan!")
            else:
                st.success("✅ Clear: No major threats detected by the model scan threshold.")
