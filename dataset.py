import os
import random
from glob import glob
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


# Dataset
class GANDataset(Dataset):
    def __init__(self, root_dir="data", mode="train",
                 transform=None, val_ratio=0.1, seed=2026):
        """
        Folder structure:

        root_dir/
            trainA/   -> photo
            trainB/   -> painting
            testA/    -> photo
            testB/    -> painting

        mode:
            "train"  -> uses train split
            "val"    -> uses validation split (from train folder)
            "test"   -> uses test folder
        """

        super().__init__()
        self.transform = transform
        random.seed(seed)

        if mode == "test":
            self.photo_paths = sorted(
                glob(os.path.join(root_dir, "testA", "*"))
            )
            self.painting_paths = sorted(
                glob(os.path.join(root_dir, "testB", "*"))
            )

        else:
            all_photo = sorted(
                glob(os.path.join(root_dir, "trainA", "*"))
            )
            all_painting = sorted(
                glob(os.path.join(root_dir, "trainB", "*"))
            )

            random.shuffle(all_photo)
            random.shuffle(all_painting)

            val_size_photo = int(len(all_photo) * val_ratio)
            val_size_painting = int(len(all_painting) * val_ratio)

            if mode == "train":
                self.photo_paths = all_photo[val_size_photo:]
                self.painting_paths = all_painting[val_size_painting:]
            elif mode == "val":
                self.photo_paths = all_photo[:val_size_photo]
                self.painting_paths = all_painting[:val_size_painting]
            else:
                raise ValueError("mode must be train, val, or test")

    def __len__(self):
        return max(len(self.photo_paths),
                   len(self.painting_paths))

    def __getitem__(self, index):
        photo_path = random.choice(self.photo_paths)
        painting_path = random.choice(self.painting_paths)

        photo_img = Image.open(photo_path).convert("RGB")
        painting_img = Image.open(painting_path).convert("RGB")

        if self.transform:
            photo_img = self.transform(photo_img)
            painting_img = self.transform(painting_img)

        return photo_img, painting_img


# Transforms
def get_train_transforms():
    return transforms.Compose([
        transforms.Resize(286),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])


def get_test_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])



# Main (for debug only)
if __name__ == "__main__":
    root = "data"

    train_dataset = GANDataset(
        root_dir=root,
        mode="train",
        transform=get_train_transforms()
    )

    val_dataset = GANDataset(
        root_dir=root,
        mode="val",
        transform=get_test_transforms()
    )

    test_dataset = GANDataset(
        root_dir=root,
        mode="test",
        transform=get_test_transforms()
    )

    print("Dataset sizes (no files modified):")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size:   {len(val_dataset)}")
    print(f"Test size:  {len(test_dataset)}")

    photo, painting = train_dataset[0]
    print("Sample tensor shapes:", photo.shape, painting.shape)