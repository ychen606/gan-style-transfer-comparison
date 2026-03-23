import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from cyclegan_model import CycleGAN
from dataset import GANDataset, get_test_transforms


# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = os.path.join("cyclegan_models", "cyclegan_50.pth")
BATCH_SIZE = 1


# Output Folders
BASE_OUTPUT_DIR = "cyclegan_outputs"
PHOTO_TO_PAINTING_DIR = os.path.join(BASE_OUTPUT_DIR, "photo_to_painting")
PAINTING_TO_PHOTO_DIR = os.path.join(BASE_OUTPUT_DIR, "painting_to_photo")

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
os.makedirs(PHOTO_TO_PAINTING_DIR, exist_ok=True)
os.makedirs(PAINTING_TO_PHOTO_DIR, exist_ok=True)


# Load Model
model = CycleGAN().to(DEVICE)
model.eval()

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

model.G_photo_to_painting.load_state_dict(checkpoint["G_photo_to_painting"])
model.G_painting_to_photo.load_state_dict(checkpoint["G_painting_to_photo"])

print("Model loaded successfully.")


# Load Test Dataset
test_dataset = GANDataset(
    root_dir="data",
    mode="test",
    transform=get_test_transforms()
)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Run Testing
with torch.no_grad():

    for idx in range(min(len(test_dataset.photo_paths), len(test_dataset.painting_paths))):

        # Access original file paths
        photo_path = test_dataset.photo_paths[idx]
        painting_path = test_dataset.painting_paths[idx]

        photo_name = os.path.splitext(os.path.basename(photo_path))[0]
        painting_name = os.path.splitext(os.path.basename(painting_path))[0]

        real_photo, real_painting = test_dataset[idx]

        real_photo = real_photo.unsqueeze(0).to(DEVICE)
        real_painting = real_painting.unsqueeze(0).to(DEVICE)

        fake_painting = model.G_photo_to_painting(real_photo)
        fake_photo = model.G_painting_to_photo(real_painting)

        real_photo_save = (real_photo + 1) / 2
        real_painting_save = (real_painting + 1) / 2
        fake_painting_save = (fake_painting + 1) / 2
        fake_photo_save = (fake_photo + 1) / 2

        # Save photo → painting
        save_image(
            real_photo_save,
            os.path.join(PHOTO_TO_PAINTING_DIR, f"{photo_name}_input.jpg")
        )
        save_image(
            fake_painting_save,
            os.path.join(PHOTO_TO_PAINTING_DIR, f"{photo_name}_output.jpg")
        )

        # Save painting → photo
        save_image(
            real_painting_save,
            os.path.join(PAINTING_TO_PHOTO_DIR, f"{painting_name}_input.jpg")
        )
        save_image(
            fake_photo_save,
            os.path.join(PAINTING_TO_PHOTO_DIR, f"{painting_name}_output.jpg")
        )


print("Testing complete")
