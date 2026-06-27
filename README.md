# Fake Face (Deepfake) Detection System
### Computer Vision | Convolutional Neural Networks | PyTorch 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

An end-to-end Computer Vision pipeline engineered to accurately distinguish between authentic human faces and AI-generated deepfakes by identifying hidden deepfake artifacts. This project encompasses a complete MLOps lifecycle, from massive 64GB raw video frame extraction and facial cropping to multi-model CNN ensemble evaluation and interactive deployment.

---

## Cloud Infrastructure & Open-Source Hosting

To bypass local hardware constraints and GitHub file limits, the processed image datasets and raw PyTorch weights are securely hosted on the Hugging Face Hub. The Streamlit UI fetches these weights dynamically via the Hugging Face API.

* **Dataset Repository:** [subhan1501/fake-face-detection-dataset](https://huggingface.co/datasets/subhan1501/fake-face-detection-dataset)
* **Model Registry (EfficientNet & Xception):** [subhan1501/fake-face-cnn-ensemble](https://huggingface.co/subhan1501/fake-face-cnn-ensemble)

---

## Key Features & Architecture

### 1. Data Engineering Pipeline (64GB -> Cloud)
Raw video files were processed through a heavily structured extraction pipeline:
* **Frame Extraction:** Sampling sequences from raw massive media arrays.
* **Face Cropping:** Utilizing facial bounding boxes (Haar Cascades / MTCNN) to isolate features and remove background noise.
* **Stratified Splitting:** Clean isolation of train/val/test data to prevent target leakage.

### 2. Multi-Model Evaluation
Distinct Convolutional Neural Network architectures were trained and ensembled to maximize accuracy against sophisticated spoofing:
1. **EfficientNet:** Optimized for superior scaling, parameter efficiency, and granular artifact detection.
2. **Xception Net:** Leverages depthwise separable convolutions; a proven industry standard for isolating deepfake inconsistencies.

### 3. Interactive Streamlit Interface
The inference engine is wrapped in a dynamic Streamlit frontend, allowing users to upload images and receive real-time classification probability splits and a final ensemble verdict.

---

## Installation & Usage

**1. Clone the repository:**
```bash
git clone [https://github.com/subhan1501/fake-face-detection-pytorch.git](https://github.com/subhan1501/fake-face-detection-pytorch.git)
cd fake-face-detection-pytorch
```
**2. Initialize Virtual Environment & Dependencies:**

```Bash
python -m venv venv
source venv/bin/activate  # On Windows: `.\venv\Scripts\Activate.ps1`
pip install -r requirements.txt
```
**3. Launch the Application:**
(Note: Ensure you have an active internet connection on the first run so the application can dynamically download and cache the .pth weights from Hugging Face).

```Bash
python -m streamlit run app/app.py
```
## Developer Information
**Developer:** Muhammad Subhan Shahid
**Affiliation:** National University of Computer and Emerging Sciences (FAST-NU)
