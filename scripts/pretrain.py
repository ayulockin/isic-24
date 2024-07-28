import os

print(os.cpu_count())

import copy
import wandb

wandb.require("core")

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import torch.optim as optim
from torch import nn
import torch.backends.cudnn as cudnn
from torcheval.metrics.functional import binary_auroc

from torchvision.models import (
    convnext_tiny,
    convnext_small,
)

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from colorama import Fore, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")

cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

configs = dict(
    image_size = 224,
)

train_metadata_df = pd.read_csv("../data/stratified_5_fold_train_metadata.csv")

def add_path(row):
    return f"../data/train-image/image/{row.isic_id}.jpg"

train_metadata_df["path"] = train_metadata_df.apply(lambda row: add_path(row), axis=1)

cols = ["tbp_lv_symm_2axis", "tbp_lv_areaMM2", "tbp_lv_color_std_mean", "tbp_lv_norm_border", "tbp_lv_perimeterMM"]
train_metadata_df = train_metadata_df[["path", "fold"]+cols]
train_metadata_df.head()

scaler = MinMaxScaler()

train_metadata_df[cols] = scaler.fit_transform(train_metadata_df[cols])
train_metadata_df.head()


class PretrainSkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        assert "path" in df.columns
        assert "tbp_lv_symm_2axis" in df.columns
        # TODO: add more features

        self.paths = df.path.tolist()
        self.labels = df[cols].values # continuous, float
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = read_image(self.paths[idx]).to(torch.float32) / 255.0
        label = self.labels[idx]
        if self.transform:
            image = image.numpy().transpose((1,2,0))
            image = self.transform(image=image)["image"]
        return image, label

train_df = train_metadata_df.loc[
    train_metadata_df.fold != 0
]
valid_df = train_metadata_df.loc[
    train_metadata_df.fold == 0
]

transforms_train = A.Compose([
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.7),

    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.),
        A.ElasticTransform(alpha=3),
    ], p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
    A.Resize(configs["image_size"], configs["image_size"]),
    A.Normalize(
        mean=(0.6962, 0.5209, 0.4193),
        std=(0.1395, 0.1320, 0.1240)
    ),
    ToTensorV2(),
])

transforms_val = A.Compose([
    A.Resize(configs["image_size"], configs["image_size"]),
    A.Normalize(
        mean=(0.6962, 0.5209, 0.4193),
        std=(0.1395, 0.1320, 0.1240)
    ),
    ToTensorV2(),
])

num_workers = 24 # based on profiling

train_dataset = PretrainSkinDataset(train_df, transform=transforms_train)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, drop_last=True)

valid_dataset = PretrainSkinDataset(valid_df, transform=transforms_val)
valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, drop_last=True)

dataset_sizes = {"train": len(train_dataset), "val": len(valid_dataset)}
print(dataset_sizes)

imgs, labels = next(iter(train_dataloader))

model_ft = convnext_tiny()
model_ft.classifier[2] = nn.Linear(model_ft.classifier[2].in_features, 5, bias=False)

model_ft = model_ft.to(device)
model_ft = torch.compile(model_ft)
print(model_ft)


def train_model(model, dataloader, criterions, optimizer, train_step, scheduler=None):
    model.train()  # Set model to training mode

    running_loss = 0.0

    for idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device).to(torch.float32)
        # print(f"{labels.shape=}")

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            # print(f"{outputs.shape=}")
            _losses = []
            for c, criterion in enumerate(criterions):
                _losses.append(criterion(outputs[:, c], labels[:, c]))

            loss = sum(_losses)
            loss.backward()
            optimizer.step()

        loss_val = loss.detach()
        running_loss += loss_val * inputs.size(0)

        if (idx + 1) % 10 == 0:
            train_step += 1
            wandb.log({"train_loss": loss_val, "train_step": train_step})
            print(f"Train Batch Loss: {loss_val}")

    epoch_loss = running_loss / dataset_sizes["train"]

    return model, epoch_loss


def validate_model(model, dataloader, criterions, optimizer, valid_step):
    model.eval()

    running_loss = 0.0

    for idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device).to(torch.float32)

        optimizer.zero_grad()

        with torch.no_grad():
            outputs = model(inputs)
            _losses = []
            for c, criterion in enumerate(criterions):
                _losses.append(criterion(outputs[:, c], labels[:, c]))
        
        loss = sum(_losses)
        loss_val = loss.detach()
        running_loss += loss_val * inputs.size(0)

        if (idx + 1) % 10 == 0:
            valid_step += 1
            wandb.log({"valid_loss": loss_val, "valid_step": valid_step})
            print(f"valid Batch Loss: {loss_val}")

    valid_loss = running_loss / dataset_sizes["val"]

    return model, valid_loss

criterions = [nn.MSELoss()]*5
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=1e-6)

run = wandb.init(project="isic_lesions_24", job_type="pretrain")
wandb.define_metric("train_step")
wandb.define_metric("valid_step")

train_step = 0
valid_step = 0

best_epoch_auroc = -np.inf
best_valid_loss = np.inf
early_stopping_patience = 4
epochs_no_improve = 0

for epoch in range(10):
    model_ft, epoch_loss = train_model(
        model_ft, train_dataloader, criterions, optimizer_ft, train_step,
    )
    model_ft, valid_loss = validate_model(
        model_ft, valid_dataloader, criterions, optimizer_ft, valid_step
    )

    print(
        f"Epoch: {epoch} | Train Loss: {epoch_loss} | Valid Loss: {valid_loss}\n"
    )
    wandb.log(
        {
            "epoch": epoch,
            "epoch_loss": epoch_loss,
            "epoch_val_loss": valid_loss,
        }
    )

    # earlystopping dependent on validation loss
    if best_valid_loss >= valid_loss:
        print(f"{b_}Validation Loss Improved ({best_valid_loss} ---> {valid_loss}){sr_}")
        
        # checkpointing
        best_model_wts = copy.deepcopy(model_ft.state_dict())
        PATH = "../models/pretrain_valid_loss{:.4f}_epoch{:.0f}.bin".format(valid_loss, epoch)
        torch.save(model_ft.state_dict(), PATH)
        # Save a model file from the current directory
        print(f"{b_}Model Saved{sr_}")
        best_valid_loss = valid_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stopping_patience:
        print(
            f"{b_}Early stopping triggered after {epochs_no_improve} epochs with no improvement.{sr_}"
        )
        break
