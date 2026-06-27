import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
from huggingface_hub import hf_hub_download

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

REPO_ID = "subhan1501/fake-face-cnn-ensemble"
EFF_FILE = "efficientnet.pth"
XCP_FILE = "xception.pth"

st.set_page_config(page_title="Fake Face Detection System", layout="centered", page_icon="🎭")

@st.cache_resource
def load_models():
    eff_path = hf_hub_download(repo_id=REPO_ID, filename=EFF_FILE)
    xcp_path = hf_hub_download(repo_id=REPO_ID, filename=XCP_FILE)

    # Initialize architectures and load weights
    eff_model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    eff_model.load_state_dict(torch.load(eff_path, map_location=DEVICE))
    eff_model.to(DEVICE).eval()

    xcp_model = timm.create_model("xception", pretrained=False, num_classes=1)
    xcp_model.load_state_dict(torch.load(xcp_path, map_location=DEVICE))
    xcp_model.to(DEVICE).eval()

    return eff_model, xcp_model

with st.spinner("🔄 Fetching cloud models and initializing architectures..."):
    eff_model, xcp_model = load_models()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

st.title("🎭 Fake Face Detection System")
st.write("This application detects **real vs. fake (deepfake) faces** using an **ensemble of EfficientNet and Xception** deep learning architectures.")
st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 🖼️ Uploaded Image")
        st.image(image, use_container_width=True)
    
    img = transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        eff_prob = torch.sigmoid(eff_model(img)).item()
        xcp_prob = torch.sigmoid(xcp_model(img)).item()
    
    ensemble_prob = (eff_prob + xcp_prob) / 2
    
    with col2:
        st.markdown("### 📊 Model Confidence Scores")
        sub_col1, sub_col2 = st.columns(2)
        sub_col1.metric("EfficientNet", f"{eff_prob*100:.2f}% Fake")
        sub_col2.metric("XceptionNet", f"{xcp_prob*100:.2f}% Fake")
        
        st.write("**Ensemble Risk Level:**")
        st.progress(min(ensemble_prob, 1.0))
        
        st.markdown("### 🏁 Ensemble Decision")
        if ensemble_prob > 0.5:
            st.error(f"🚨 **FAKE FACE DETECTED** \nCalculated Confidence: **{ensemble_prob*100:.2f}%**")
        else:
            st.success(f"✅ **AUTHENTIC REAL FACE** \nCalculated Confidence: **{(1-ensemble_prob)*100:.2f}%**")

st.markdown("---")
st.caption("© 2026 | Fake Face Detection Architecture | Muhammad Subhan Shahid")