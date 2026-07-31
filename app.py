import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Set up page configurations
st.set_page_config(
    page_title="Weapon Detection App",
    page_icon="🛡️",
    layout="wide"
)

# 1. Load the fine-tuned Weapon Detection model weights safely using caching
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    new_model = load_model()
except Exception as e:
    st.error(f"Error loading model weights: {e}. Please make sure 'best.pt' is in your application directory.")
    st.stop()

# 2. Main Sidebar Configuration
st.sidebar.header("🔧 Model Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.01, 
    max_value=1.00, 
    value=0.25, 
    step=0.01,
    help="Lower this value if the model fails to detect obvious weapons in the image."
)

# 3. Main User Interface Header
st.title("🛡️ Weapon Detection App")
st.write("Upload an X-ray scan to detect hidden threats or weapons.")

# 4. File Uploader Widget
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png", "jpeg"])

# 5. Processing Pipeline
if uploaded_file:
    input_image = Image.open(uploaded_file)

    # Establish side-by-side columns matching the user mockup interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(input_image, use_container_width=True)
        
    with col2:
        st.subheader("Object Detection")
        
        # Action button to trigger inference pipeline execution
        if st.button("Run Threat Scan", use_container_width=True):
            with st.spinner("Analyzing image for weapons..."):
                # Run model prediction using the validated threshold from the sidebar slider
                prediction_list = new_model(input_image, conf=conf_threshold)       
            
            # FIX: Index the list at position [0] to extract the explicit Results item
            results = prediction_list[0]
            
            # Now .plot() will successfully render the bounding boxes and text labels
            annotated_image = results.plot()
            
            st.image(
                annotated_image,  
                use_container_width=True, 
                caption=f"Scan Analysis Complete (Confidence Threshold: {conf_threshold})"
            )
            
            # Display detailed alert status
            detection_count = len(results.boxes)
            if detection_count > 0:
                st.warning(f"⚠️ Warning: Detected {detection_count} potential threat(s) in the scan!")
                
                # Print out details of each box detected
                for box in results.boxes:
                    class_id = int(box.cls)
                    class_name = new_model.names[class_id]
                    confidence = float(box.conf)
                    st.write(f"- Found **{class_name}** with **{confidence:.2%}** confidence.")
            else:
                st.success("✅ No threats detected at the current confidence threshold level. Try lowering the slider in the sidebar.")
