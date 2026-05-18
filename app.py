import streamlit as st
import torch
import torch.nn as nn

from torchvision import transforms
from PIL import Image

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="FairVision",
    page_icon="📷",
    layout="centered"
)

# -----------------------------
# TITLE
# -----------------------------

st.title("FairVision")

st.subheader(
    "CNN-Based Age Group Classification System"
)

# -----------------------------
# AGE LABELS
# -----------------------------

age_labels = [
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+"
]

# -----------------------------
# DEVICE
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------
# TRANSFORM
# -----------------------------

test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5]
    )
])

# -----------------------------
# CNN MODEL
# -----------------------------

class AgeCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 9)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

# -----------------------------
# LOAD MODEL
# -----------------------------

try:

    model = AgeCNN().to(device)

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=device
        )
    )

    model.eval()

    st.success("Model loaded successfully!")

except Exception as e:

    st.error(f"Error loading model: {e}")

    st.stop()

# -----------------------------
# FILE UPLOADER
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PREDICTION
# -----------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    input_image = test_transform(image)

    input_image = input_image.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(input_image)

        probabilities = torch.softmax(outputs, dim=1)

        top3_prob, top3_idx = torch.topk(probabilities, 3)

    st.subheader("Top 3 Predictions")

    for i in range(3):

        label = age_labels[top3_idx[0][i].item()]

        confidence = top3_prob[0][i].item() * 100

        st.write(
            f"{i+1}. {label} : {confidence:.2f}%"
        )

# -----------------------------
# LIMITATIONS
# -----------------------------

st.markdown("---")

st.subheader("Responsible AI & Limitations")

st.write(
    "This model was trained using the FairFace dataset for educational purposes. "
    "Predictions may contain errors or demographic bias. "
    "The system should not be used for high-risk or real-world biometric decisions."
)