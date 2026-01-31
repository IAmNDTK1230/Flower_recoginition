# IMPORT 
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import json
import tensorflow as tf
import keras

# CONFIG 
st.set_page_config(
    page_title="Flower Detection & Classification",
    layout="wide"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 102

# PATHS 
CNN_PATH = r"C:\Users\ASUS\\Project1\models\best_convnext.pth"
VIT_PATH = r"C:\Users\ASUS\Project1\models\best_vit.keras"
YOLO_PATH = r"C:\Users\ASUS\Project1\models\best_yolo.pt"

CAT_TO_NAME_PATH = "cat_to_name.json"
CLASS_TO_IDX_PATH = "class_to_idx.json"

# BUILD CONVNEXT
def build_convnext_base(num_classes):
    model = models.convnext_base(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    model.to(DEVICE)
    model.eval()
    return model

# LOAD LABELS
with open(CAT_TO_NAME_PATH, "r", encoding="utf-8") as f:
    folder_to_name = json.load(f)

with open(CLASS_TO_IDX_PATH, "r") as f:
    folder_to_idx = json.load(f)

idx_to_folder = {int(idx): folder for folder, idx in folder_to_idx.items()}
idx_to_name = {idx: folder_to_name[folder] for idx, folder in idx_to_folder.items()}

# TRANSFORM (ConvNeXt)
cnn_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ViT PREPROCESS (Keras) 
def vit_preprocess(img_pil):
    img = img_pil.resize((224, 224))
    img = np.array(img).astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# LOAD MODELS 
@st.cache_resource
def load_convnext():
    model = build_convnext_base(NUM_CLASSES)
    ckpt = torch.load(CNN_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    return model

@st.cache_resource
def load_vit():
    import keras_hub
    model = keras.models.load_model(VIT_PATH)
    return model


@st.cache_resource
def load_yolo():
    return YOLO(YOLO_PATH)

cnn_model = load_convnext()
vit_model = load_vit()
yolo_model = load_yolo()

# CLASSIFY FUNCTIONS 
def classify_flower_cnn(img_pil):
    x = cnn_transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = cnn_model(x)
        probs = F.softmax(logits, dim=1)

    idx = probs.argmax(1).item()
    conf = probs.max().item() * 100
    return idx_to_name[idx], conf


def classify_flower_vit(img_pil):
    x = vit_preprocess(img_pil)
    preds = vit_model.predict(x, verbose=0)

    idx = int(np.argmax(preds[0]))
    conf = float(preds[0][idx]) * 100
    return idx_to_name[idx], conf

# YOLO + CLASSIFIER PIPELINE
def detect_and_classify(image_pil, model_choice):
    img_np = np.array(image_pil)

    results = yolo_model.predict(
        source=img_np,
        conf=0.25,
        device=0 if DEVICE == "cuda" else "cpu",
        verbose=False
    )

    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if (x2 - x1) < 40 or (y2 - y1) < 40:
            continue

        crop = image_pil.crop((x1, y1, x2, y2))

        if model_choice == "ConvNeXt":
            name, conf = classify_flower_cnn(crop)
        else:
            name, conf = classify_flower_vit(crop)

        detections.append((x1, y1, x2, y2, name, conf))

    return detections

# STREAMLIT UI
st.title("Flower Detection & Classification")

st.sidebar.header("⚙️ Model Settings")
model_choice = st.sidebar.radio(
    "Choose classifier model:",
    ["ConvNeXt", "ViT"]
)

uploaded = st.file_uploader(
    "Upload ảnh có nhiều hoa",
    type=["jpg", "png", "jpeg"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    detections = detect_and_classify(image, model_choice)

    img_draw = np.array(image)
    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)

    for x1, y1, x2, y2, name, conf in detections:
        label = f"{name} ({conf:.1f}%)"
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_draw,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns([1.3, 1])

    col1.image(
        img_draw,
        caption=f"Detection Result using {model_choice}",
        use_container_width=True
    )

    with col2:
        st.subheader("Detected Flowers")

        if len(detections) == 0:
            st.warning("Không phát hiện hoa")
        else:
            # Gom hoa trùng nhau, lấy confidence cao nhất
            best_flowers = {}

            for _, _, _, _, name, conf in detections:
                if name not in best_flowers:
                    best_flowers[name] = conf
                else:
                    best_flowers[name] = max(best_flowers[name], conf)

            # Hiển thị
            for i, (name, conf) in enumerate(best_flowers.items()):
                display_name = name.title()  # viết hoa chữ cái đầu

                st.markdown(f"### Flower {i+1}: {display_name}")
                st.progress(conf / 100)
                st.caption(f"{conf:.2f}%")
                st.divider()

