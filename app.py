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

st.sidebar.header("🔧 Model Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.01, 
    max_value=1.00, 
    value=0.25, 
    step=0.01,
    help="Lower this value if the model fails to detect obvious weapons in the image."
)

st.title("🛡️ Weapon Detection App")
st.write("Upload an X-ray scan to detect hidden threats or weapons.")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    input_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)
        
    with col2:
        st.subheader("Object Detection")
        
        if st.button("Predict", use_container_width=True):
            with st.spinner("Analyzing image for weapons..."):
                results = new_model(input_image, conf=conf_threshold)       
            
            annotated_image = results.plot()
            
            st.image(
                annotated_image,  
                use_container_width=True, 
                caption=f"Scan Analysis Complete (Confidence Threshold: {conf_threshold})"
            )
            
            detection_count = len(results.boxes)
            if detection_count > 0:
                st.warning(f"⚠️ Warning: Detected {detection_count} potential threat(s) in the scan!")
                
                for box in results.boxes:
                    class_id = int(box.cls[0])
                    class_name = new_model.names[class_id]
                    confidence = float(box.conf[0])
                    st.write(f"- Found **{class_name}** with **{confidence:.2%}** confidence.")
            else:
                st.success("✅ No threats detected at the current confidence threshold level. Try lowering the slider in the sidebar.")
