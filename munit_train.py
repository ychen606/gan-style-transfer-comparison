import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from munit_model import MUNIT
from dataset import GANDataset, get_train_transforms, get_test_transforms


# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 1
LR = 1e-4
EPOCHS = 70

WEIGHT_RECON_I = 10      # image reconstruction
WEIGHT_RECON_C = 1       # content reconstruction
WEIGHT_RECON_S = 1       # style reconstruction
WEIGHT_ADV = 1

SAVE_DIR = "munit_models"
os.makedirs(SAVE_DIR, exist_ok=True)

RESUME_PATH = None
#RESUME_PATH = os.path.join(SAVE_DIR, "munit_60.pth")
START_EPOCH = 0


# Initialize model
model = MUNIT().to(DEVICE)

G_params = list(model.G_photo.parameters()) + \
           list(model.G_painting.parameters())

optimizer_G = optim.Adam(G_params, lr=LR, betas=(0.5, 0.999))
optimizer_D_photo = optim.Adam(model.D_photo.parameters(), lr=LR, betas=(0.5, 0.999))
optimizer_D_painting = optim.Adam(model.D_painting.parameters(), lr=LR, betas=(0.5, 0.999))


# Resume training
if RESUME_PATH is not None and os.path.exists(RESUME_PATH):
    checkpoint = torch.load(RESUME_PATH, map_location=DEVICE)

    model.G_photo.load_state_dict(checkpoint['G_photo'])
    model.G_painting.load_state_dict(checkpoint['G_painting'])
    model.D_photo.load_state_dict(checkpoint['D_photo'])
    model.D_painting.load_state_dict(checkpoint['D_painting'])

    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D_photo.load_state_dict(checkpoint['optimizer_D_photo'])
    optimizer_D_painting.load_state_dict(checkpoint['optimizer_D_painting'])

    START_EPOCH = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {START_EPOCH}")
else:
    print("Starting training from scratch")


# Loss functions
criterion_adv = nn.MSELoss()
criterion_recon = nn.L1Loss()


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
for epoch in range(START_EPOCH, EPOCHS):

    model.train()

    total_G_loss = 0
    total_D_loss = 0

    for real_photo, real_painting in train_loader:

        real_photo = real_photo.to(DEVICE)
        real_painting = real_painting.to(DEVICE)

        # Encode
        c_photo, s_photo = model.G_photo.encode(real_photo)
        c_painting, s_painting = model.G_painting.encode(real_painting)

        # Random styles
        s_photo_rand = torch.randn_like(s_photo)
        s_painting_rand = torch.randn_like(s_painting)

        # Decode (cross-domain)
        fake_painting = model.G_painting.decode(c_photo, s_painting_rand)
        fake_photo = model.G_photo.decode(c_painting, s_photo_rand)

        # Reconstruction (same domain)
        rec_photo = model.G_photo.decode(c_photo, s_photo)
        rec_painting = model.G_painting.decode(c_painting, s_painting)

        # Re-encode fakes (for latent reconstruction)
        c_photo_recon, s_painting_recon = model.G_painting.encode(fake_painting)
        c_painting_recon, s_photo_recon = model.G_photo.encode(fake_photo)


        # Train Generators
        optimizer_G.zero_grad()

        # Adversarial loss
        pred_fake_painting = model.D_painting(fake_painting)
        loss_adv_painting = criterion_adv(pred_fake_painting, torch.ones_like(pred_fake_painting))

        pred_fake_photo = model.D_photo(fake_photo)
        loss_adv_photo = criterion_adv(pred_fake_photo, torch.ones_like(pred_fake_photo))

        loss_adv = loss_adv_painting + loss_adv_photo

        # Image reconstruction loss
        loss_recon_i = criterion_recon(rec_photo, real_photo) + \
                       criterion_recon(rec_painting, real_painting)

        # Content reconstruction loss
        loss_recon_c = criterion_recon(c_photo_recon, c_photo) + \
                       criterion_recon(c_painting_recon, c_painting)

        # Style reconstruction loss
        loss_recon_s = criterion_recon(s_painting_recon, s_painting_rand) + \
                       criterion_recon(s_photo_recon, s_photo_rand)

        # Total generator loss
        loss_G = WEIGHT_ADV * loss_adv + \
                 WEIGHT_RECON_I * loss_recon_i + \
                 WEIGHT_RECON_C * loss_recon_c + \
                 WEIGHT_RECON_S * loss_recon_s

        loss_G.backward()
        optimizer_G.step()


        # Train Discriminators

        # Photo discriminator
        optimizer_D_photo.zero_grad()

        pred_real = model.D_photo(real_photo)
        loss_real = criterion_adv(pred_real, torch.ones_like(pred_real))

        pred_fake = model.D_photo(fake_photo.detach())
        loss_fake = criterion_adv(pred_fake, torch.zeros_like(pred_fake))

        loss_D_photo = 0.5 * (loss_real + loss_fake)
        loss_D_photo.backward()
        optimizer_D_photo.step()

        # Painting discriminator
        optimizer_D_painting.zero_grad()

        pred_real = model.D_painting(real_painting)
        loss_real = criterion_adv(pred_real, torch.ones_like(pred_real))

        pred_fake = model.D_painting(fake_painting.detach())
        loss_fake = criterion_adv(pred_fake, torch.zeros_like(pred_fake))

        loss_D_painting = 0.5 * (loss_real + loss_fake)
        loss_D_painting.backward()
        optimizer_D_painting.step()

        loss_D = loss_D_photo + loss_D_painting

        total_G_loss += loss_G.item()
        total_D_loss += loss_D.item()


    # Validation
    model.eval()
    val_recon_loss = 0

    with torch.no_grad():
        for real_photo, real_painting in val_loader:

            real_photo = real_photo.to(DEVICE)
            real_painting = real_painting.to(DEVICE)

            c_photo, s_photo = model.G_photo.encode(real_photo)
            rec_photo = model.G_photo.decode(c_photo, s_photo)

            c_painting, s_painting = model.G_painting.encode(real_painting)
            rec_painting = model.G_painting.decode(c_painting, s_painting)

            val_recon_loss += (
                criterion_recon(rec_photo, real_photo) +
                criterion_recon(rec_painting, real_painting)
            ).item()

    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    print(f"  Generator Loss: {total_G_loss/len(train_loader):.4f}")
    print(f"  Discriminator Loss: {total_D_loss/len(train_loader):.4f}")
    print(f"  Val Recon Loss: {val_recon_loss/len(val_loader):.4f}")

    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        save_path = os.path.join(SAVE_DIR, f"munit_{epoch+1}.pth")

        checkpoint = {
            'epoch': epoch,
            'G_photo': model.G_photo.state_dict(),
            'G_painting': model.G_painting.state_dict(),
            'D_photo': model.D_photo.state_dict(),
            'D_painting': model.D_painting.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D_photo': optimizer_D_photo.state_dict(),
            'optimizer_D_painting': optimizer_D_painting.state_dict()
        }

        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved at {save_path}")

    print("-" * 30)