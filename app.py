import streamlit as st
from rembg import remove
from PIL import Image, ImageEnhance, ImageOps
from streamlit_cropper import st_cropper
import numpy as np
import cv2
import io

st.set_page_config(
    page_title="AI Photo Studio",
    page_icon="🎨",
    layout="wide"
)

st.markdown("""
<style>
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
.main {
    background: linear-gradient(120deg, #0f2027, #203a43, #2c5364);
    color: white;
    animation: fadeIn 0.8s ease-in-out;
}
.stSidebar {
    background: rgba(15, 15, 15, 0.95);
}
.card {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    margin-bottom: 20px;
}
.stDownloadButton button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color: white;
    border-radius: 12px;
    padding: 12px 22px;
    font-weight: bold;
    border: none;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_remove_bg(img_bytes):
    """Caches the background removal based on image bytes."""
    return remove(img_bytes)

@st.cache_data(show_spinner=False)
def ai_image_enhancer(image):
    """Caches the heavy AI denoising and sharpening."""
    img = np.array(image.convert("RGB"))
    
    img = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)
    
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    return Image.fromarray(img)

def apply_advanced_filter(image, filter_type):
    if filter_type == "None":
        return image
    if filter_type == "Posterize":
        return ImageOps.posterize(image.convert("RGB"), 4)
    if filter_type == "Solarize":
        return ImageOps.solarize(image.convert("RGB"), threshold=128)
    if filter_type == "Retro":
        img = ImageEnhance.Color(image).enhance(1.6)
        return ImageEnhance.Contrast(img).enhance(1.2)
    if filter_type == "Sketch":
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return Image.fromarray(sketch)
    return image

st.markdown("<h1 style='text-align:center;'>🎨 AI Photo Studio</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Optimized for Streamlit Cloud</p>", unsafe_allow_html=True)
st.divider()

uploaded_file = st.sidebar.file_uploader("📤 Upload Image", ["jpg", "jpeg", "png"])

if uploaded_file:
    raw_image = Image.open(uploaded_file).convert("RGBA")
    
    tab1, tab2, tab3 = st.sidebar.tabs(["✂️ Crop & BG", "🧠 Effects", "⚙️ Frame"])

    with tab1:
        do_crop = st.checkbox("Enable Cropping")
        bg_mode = st.radio("Background Mode", ["Original", "Remove BG", "Color Fill", "Custom Image"])
        bg_color = st.color_picker("Fill Color", "#ffffff") if bg_mode == "Color Fill" else None
        bg_upload = st.file_uploader("Upload BG", ["jpg", "png"]) if bg_mode == "Custom Image" else None

    with tab2:
        filter_option = st.selectbox("Filter", ["None", "Posterize", "Solarize", "Retro", "Sketch"])
        enhance_ai = st.toggle("Activate AI Enhancement")

    with tab3:
        frame_width = st.slider("Frame Width", 0, 60, 0)
        frame_color = st.color_picker("Frame Color", "#000000")

    
  
    work_img = raw_image
    if do_crop:
        work_img = st_cropper(work_img, realtime_update=True)

   
    if bg_mode != "Original":
       
        img_byte_arr = io.BytesIO()
        work_img.save(img_byte_arr, format='PNG')
        
        with st.spinner("AI Background Removal..."):
            subject_bytes = get_remove_bg(img_byte_arr.getvalue())
            subject = Image.open(io.BytesIO(subject_bytes)).convert("RGBA")

        if bg_mode == "Remove BG":
            work_img = subject
        elif bg_mode == "Color Fill":
            bg = Image.new("RGBA", subject.size, bg_color)
            work_img = Image.alpha_composite(bg, subject)
        elif bg_mode == "Custom Image" and bg_upload:
            bg_img = Image.open(bg_upload).convert("RGBA").resize(subject.size)
            work_img = Image.alpha_composite(bg_img, subject)

   
    if enhance_ai:
        work_img = ai_image_enhancer(work_img)

    work_img = apply_advanced_filter(work_img, filter_option)
    
    if frame_width > 0:
        work_img = ImageOps.expand(work_img, frame_width, frame_color)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'><h3>Original</h3></div>", unsafe_allow_html=True)
        st.image(raw_image, use_container_width=True)

    with col2:
        st.markdown("<div class='card'><h3>Final Output</h3></div>", unsafe_allow_html=True)
        st.image(work_img, use_container_width=True)

        buf = io.BytesIO()
        work_img.convert("RGB").save(buf, format="PNG")
        st.download_button(
            "📥 Download Image",
            buf.getvalue(),
            "ai_photo_studio.png",
            "image/png"
        )
else:
    st.info("Please upload an image in the sidebar to begin.")