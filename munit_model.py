import torch
import torch.nn as nn


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


# Content Encoder
class ContentEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.ReflectionPad2d(padding=3),

            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )

    def forward(self, x):
        return self.model(x)


# Style Encoder
class StyleEncoder(nn.Module):
    def __init__(self, style_dim=8):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(output_size=1)
        )

        self.fc = nn.Linear(256, style_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# AdaIN
class AdaIN(nn.Module):
    def __init__(self, channels, style_dim):
        super().__init__()
        self.fc_gamma = nn.Linear(style_dim, channels)
        self.fc_beta = nn.Linear(style_dim, channels)

    def forward(self, x, style):
        gamma = self.fc_gamma(style).unsqueeze(2).unsqueeze(3)
        beta  = self.fc_beta(style).unsqueeze(2).unsqueeze(3)

        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-5

        x = (x - mean) / std
        return gamma * x + beta


# Decoder
class Decoder(nn.Module):
    def __init__(self, style_dim=8):
        super().__init__()

        self.adain1 = AdaIN(256, style_dim)
        self.adain2 = AdaIN(256, style_dim)

        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, content, style):
        x = self.adain1(content, style)
        x = self.res1(x)

        x = self.adain2(x, style)
        x = self.res2(x)

        return self.upsample(x)


# Generator
class Generator(nn.Module):
    def __init__(self, style_dim=8):
        super().__init__()

        self.content_encoder = ContentEncoder()
        self.style_encoder = StyleEncoder(style_dim)
        self.decoder = Decoder(style_dim)

    def encode(self, x):
        content = self.content_encoder(x)
        style = self.style_encoder(x)
        return content, style

    def decode(self, content, style):
        return self.decoder(content, style)

    def forward(self, x):
        content, style = self.encode(x)
        return self.decode(content, style)


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


# Full MUNIT
class MUNIT(nn.Module):
    def __init__(self, style_dim=8):
        super().__init__()

        self.G_photo = Generator(style_dim)
        self.G_painting = Generator(style_dim)

        self.D_photo = Discriminator()
        self.D_painting = Discriminator()

    def forward(self, real_photo, real_painting):
        content_photo, style_photo = self.G_photo.encode(real_photo)
        content_painting, style_painting = self.G_painting.encode(real_painting)

        style_photo_rand = torch.randn_like(style_photo)
        style_painting_rand = torch.randn_like(style_painting)

        fake_painting = self.G_painting.decode(content_photo, style_painting_rand)
        fake_photo = self.G_photo.decode(content_painting, style_photo_rand)

        rec_photo = self.G_photo.decode(content_photo, style_photo)
        rec_painting = self.G_painting.decode(content_painting, style_painting)

        return fake_photo, fake_painting, rec_photo, rec_painting