import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from cyclegan_model import CycleGAN
from dataset import GANDataset, get_train_transforms, get_test_transforms


# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1
LR = 2e-4
EPOCHS = 100

WEIGHT_CYCLE = 10
WEIGHT_IDENTITY = 5

SAVE_DIR = "cyclegan_models"
os.makedirs(SAVE_DIR, exist_ok=True)


# Initialize model
model = CycleGAN().to(DEVICE)

G_params = list(model.G_photo_to_painting.parameters()) + \
           list(model.G_painting_to_photo.parameters())

D_photo_params = model.D_photo.parameters()
D_painting_params = model.D_painting.parameters()

optimizer_G = optim.Adam(G_params, lr=LR, betas=(0.5, 0.999))
optimizer_D_photo = optim.Adam(D_photo_params, lr=LR, betas=(0.5, 0.999))
optimizer_D_painting = optim.Adam(D_painting_params, lr=LR, betas=(0.5, 0.999))


# Loss functions
criterion_adversarial = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()


# Data
train_dataset = GANDataset(
    root_dir="data",
    mode="train",
    transform=get_train_transforms()
)

val_dataset = GANDataset(
    root_dir="data",
    mode="val",
    transform=get_test_transforms()
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)



# Training Loop
for epoch in range(EPOCHS):

    model.train()

    total_G_loss = 0
    total_D_loss = 0

    for real_photo, real_painting in train_loader:

        real_photo = real_photo.to(DEVICE)
        real_painting = real_painting.to(DEVICE)

        # Train Generators
        optimizer_G.zero_grad()

        fake_photo, fake_painting, rec_photo, rec_painting = model(
            real_photo, real_painting
        )

        # Adversarial loss
        pred_fake_painting = model.D_painting(fake_painting)
        real = torch.ones_like(pred_fake_painting)
        loss_adversarial_photo_to_painting = criterion_adversarial(pred_fake_painting, real)

        pred_fake_photo = model.D_photo(fake_photo)
        real = torch.ones_like(pred_fake_photo)
        loss_adversarial_painting_to_photo = criterion_adversarial(pred_fake_photo, real)

        loss_adversarial = loss_adversarial_photo_to_painting + loss_adversarial_painting_to_photo

        # Cycle loss
        loss_cycle_photo = criterion_cycle(rec_photo, real_photo)
        loss_cycle_painting = criterion_cycle(rec_painting, real_painting)

        loss_cycle = loss_cycle_photo + loss_cycle_painting

        # Identity loss
        same_photo = model.G_painting_to_photo(real_photo)
        loss_identity_photo = criterion_identity(same_photo, real_photo)

        same_painting = model.G_photo_to_painting(real_painting)
        loss_identity_painting = criterion_identity(same_painting, real_painting)

        loss_identity = loss_identity_photo + loss_identity_painting

        # Total generator loss
        loss_G = loss_adversarial + \
                 WEIGHT_CYCLE * loss_cycle + \
                 WEIGHT_IDENTITY * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator for Photo
        optimizer_D_photo.zero_grad()

        pred_real = model.D_photo(real_photo)
        real = torch.ones_like(pred_real)
        loss_real = criterion_adversarial(pred_real, real)

        pred_fake = model.D_photo(fake_photo.detach())
        fake = torch.zeros_like(pred_fake)
        loss_fake = criterion_adversarial(pred_fake, fake)

        loss_D_photo = 0.5 * (loss_real + loss_fake)

        loss_D_photo.backward()
        optimizer_D_photo.step()

        # Train Discriminator for Painting
        optimizer_D_painting.zero_grad()

        pred_real = model.D_painting(real_painting)
        real = torch.ones_like(pred_real)
        loss_real = criterion_adversarial(pred_real, real)

        pred_fake = model.D_painting(fake_painting.detach())
        fake = torch.zeros_like(pred_fake)
        loss_fake = criterion_adversarial(pred_fake, fake)

        loss_D_painting = 0.5 * (loss_real + loss_fake)

        loss_D_painting.backward()
        optimizer_D_painting.step()

        loss_D = loss_D_photo + loss_D_painting

        total_G_loss += loss_G.item()
        total_D_loss += loss_D.item()


    # Validation
    model.eval()
    val_cycle_loss = 0

    with torch.no_grad():
        for real_photo, real_painting in val_loader:

            real_photo = real_photo.to(DEVICE)
            real_painting = real_painting.to(DEVICE)

            fake_photo, fake_painting, rec_photo, rec_painting = model(
                real_photo, real_painting
            )

            loss_cycle_photo = criterion_cycle(rec_photo, real_photo)
            loss_cycle_painting = criterion_cycle(rec_painting, real_painting)

            val_cycle_loss += (loss_cycle_photo + loss_cycle_painting).item()

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"  Generator Loss: {total_G_loss/len(train_loader):.4f}")
    print(f"  Discriminator Loss: {total_D_loss/len(train_loader):.4f}")
    print(f"  Val Cycle Loss: {val_cycle_loss/len(val_loader):.4f}")
    print("-" * 30)

    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        save_path = os.path.join(
            SAVE_DIR,
            f"cyclegan_{epoch+1}.pth"
        )
        torch.save(model.state_dict(), save_path)
