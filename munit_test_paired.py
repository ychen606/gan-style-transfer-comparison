import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from munit_model import MUNIT


# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = os.path.join("munit_models", "munit_70.pth")
INPUT_DIR = os.path.join("data", "test_pairs")
OUTPUT_DIR = os.path.join("munit_outputs", "pairs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Hardcoded pairs
PAIRS = [
    ("00000292_(5).jpg", "00100.jpg"),
    ("00000306_(6).jpg", "mount-kolsaas.jpg"),
    ("00000304_(3).jpg", "zaandam-1.jpg"),
    ("00000329_(4).jpg", "waves-and-rocks-at-pourville.jpg")
]


# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Load Model
model = MUNIT().to(DEVICE)
model.eval()

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.G_photo.load_state_dict(checkpoint["G_photo"])
model.G_painting.load_state_dict(checkpoint["G_painting"])

print("Model loaded successfully.")


# Testing
with torch.no_grad():

    for idx, (photo_name, painting_name) in enumerate(PAIRS):

        # Load images
        photo_path = os.path.join(INPUT_DIR, photo_name)
        painting_path = os.path.join(INPUT_DIR, painting_name)
        photo = Image.open(photo_path).convert("RGB")
        painting = Image.open(painting_path).convert("RGB")

        photo_tensor = transform(photo).unsqueeze(0).to(DEVICE)
        painting_tensor = transform(painting).unsqueeze(0).to(DEVICE)

        # Encode
        content_photo, _ = model.G_photo.encode(photo_tensor)
        _, style_painting = model.G_painting.encode(painting_tensor)

        # Decode
        fake_painting = model.G_painting.decode(content_photo, style_painting)

        # Save images
        photo_save = (photo_tensor + 1) / 2
        painting_save = (painting_tensor + 1) / 2
        fake_save = (fake_painting + 1) / 2

        base_name = f"pair_{idx}"

        save_image(photo_save, os.path.join(OUTPUT_DIR, f"{base_name}_photo_{photo_name}.jpg"))
        save_image(painting_save, os.path.join(OUTPUT_DIR, f"{base_name}_painting_{painting_name}.jpg"))
        save_image(fake_save, os.path.join(OUTPUT_DIR, f"{base_name}_output.jpg"))

        print(f"Saved pair {idx}")


print("Testing complete")