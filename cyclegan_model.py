import torch
import torch.nn as nn


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


# Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            # Reflection padding (40 x 40)
            nn.ReflectionPad2d(40),

            # 32 x 9 x 9 conv, stride 1
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 64 x 3 x 3 conv, stride 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 128 x 3 x 3 conv, stride 2
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 5 residual blocks (128 filters)
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),

            # 64 x 3 x 3 conv, stride 1/2
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 32 x 3 x 3 conv, stride 1/2
            nn.ConvTranspose2d(64, 32, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 3 x 9 x 9 conv, stride 1
            nn.Conv2d(32, 3, kernel_size=9, stride=1, padding=4)
        )

    def forward(self, x):
        return self.model(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        model = [
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]

        in_features = 64
        out_features = in_features * 2

        for n in range(1, 3):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=4,
                          stride=2 if n < 2 else 1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(0.2, True)
            ]
            in_features = out_features
            out_features = in_features * 2

        model += [
            nn.Conv2d(in_features, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# CycleGAN Model
class CycleGAN(nn.Module):
    def __init__(self):
        super().__init__()

        # Generators
        self.G_photo_to_painting = Generator()
        self.G_painting_to_photo = Generator()

        # Discriminators
        self.D_photo = Discriminator()
        self.D_painting = Discriminator()

    def forward(self, real_photo, real_painting):
        fake_painting = self.G_photo_to_painting(real_photo)
        fake_photo = self.G_painting_to_photo(real_painting)

        rec_photo = self.G_painting_to_photo(fake_painting)
        rec_painting = self.G_photo_to_painting(fake_photo)

        return fake_photo, fake_painting, rec_photo, rec_painting