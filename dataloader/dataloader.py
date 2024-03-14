from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

import config


train_transform = transforms.Compose(
    [
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

train_dataset = datasets.ImageFolder("data/train_images", transform=train_transform)
test_dataset = datasets.ImageFolder("data/test_images", transform=test_transforms)


def fetch_dataloader(types, data_dir, params):
    dataloaders = {}

    for split in ["train", "val", "test"]:
        if split in types:
            path = os.path.join(data_dir, f"{split}_images")

            if split == "train":
                dl = DataLoader(
                    train_dataset,
                    batch_size=params.batch_size,
                    shuffle=True,
                    num_workers=params.num_workers,
                    drop_last=False,
                    pin_memory=params.cuda,
                )
            else:
                dl = DataLoader(
                    test_dataset,
                    batch_size=params.batch_size,
                    shuffle=False,
                    num_workers=params.num_workers,
                    drop_last=False,
                )

            dataloaders[split] = dl

    return dataloaders
