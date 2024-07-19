## Imports

import os

print(os.cpu_count())

import copy
import wandb

wandb.require("core")

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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

## Data

train_metadata_df = pd.read_csv("../data/train-metadata.csv")
print(len(train_metadata_df))

gkf = StratifiedGroupKFold(n_splits=5)  # , shuffle=True, random_state=42

train_metadata_df["fold"] = -1
for idx, (train_idx, val_idx) in enumerate(
    gkf.split(
        train_metadata_df,
        train_metadata_df["target"],
        groups=train_metadata_df["patient_id"],
    )
):
    train_metadata_df.loc[val_idx, "fold"] = idx


def add_path(row):
    return f"../data/train-image/image/{row.isic_id}.jpg"


train_metadata_df["path"] = train_metadata_df.apply(lambda row: add_path(row), axis=1)


## Dataloader
class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, target_transform=None):
        assert "path" in df.columns
        assert "target" in df.columns

        self.paths = df.path.tolist()
        self.labels = df.target.tolist() # binary
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = read_image(self.paths[idx]).to(torch.float32) / 255.0
        label = self.labels[idx] / 1.0
        if self.transform:
            image = image.numpy().transpose((1,2,0))
            image = self.transform(image=image)["image"]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_class_samples(self, class_label):
        indices = [i for i, label in enumerate(self.labels) if label == class_label]
        return indices


data_transforms = {
    "train": A.Compose(
        [
            A.Resize(224, 224),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Downscale(p=0.25),
            A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.15, rotate_limit=60, p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ],
        p=1.0,
    ),
    "valid": A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(),
        ],
        p=1.0,
    ),
}

train_df = train_metadata_df.loc[
    train_metadata_df.fold != 0
]  # using a subset for training
valid_df = train_metadata_df.loc[
    train_metadata_df.fold == 0
]  # using another fold for validation

num_workers = 24  # based on profiling

train_dataset = SkinDataset(train_df, transform=data_transforms["train"])
neg_samples = len(train_dataset.get_class_samples(0))
pos_samples = len(train_dataset.get_class_samples(1))
neg_wts = 1 / neg_samples
pos_wts = 1 / pos_samples

sample_wts = []

for label in train_dataset.labels:
    if label == 0:
        sample_wts.append(neg_wts)
    else:
        sample_wts.append(pos_wts)

sampler = WeightedRandomSampler(
    weights=sample_wts, num_samples=len(train_dataset), replacement=True
)
train_dataloader = DataLoader(
    train_dataset,
    sampler=sampler,
    batch_size=128,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)

valid_dataset = SkinDataset(valid_df, transform=data_transforms["valid"])
valid_dataloader = DataLoader(
    valid_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
)

dataset_sizes = {"train": len(train_dataset), "val": len(valid_dataset)}

## Model

model_ft = convnext_tiny(weights="IMAGENET1K_V1")
model_ft.classifier[2] = nn.Linear(model_ft.classifier[2].in_features, 1)

model_ft = model_ft.to(device)
model_ft = torch.compile(model_ft)
print(model_ft)

## Criterion
criterion = nn.BCEWithLogitsLoss()

# Optimizer
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=1e-6)


# Train and validation functions
def train_model(model, dataloader, criterion, optimizer, scheduler=None):
    model.train()  # Set model to training mode

    running_loss = 0.0
    running_auroc = 0.0
    batch_loss = []

    for idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device).flatten()

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs).flatten()
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        loss_val = loss.detach()
        batch_loss.append(loss_val)
        running_loss += loss_val * inputs.size(0)

        auroc = binary_auroc(input=outputs, target=labels).item()
        running_auroc += auroc * inputs.size(0)

        if (idx + 1) % 100 == 0:
            wandb.log({"train_loss": loss_val, "train_auroc": auroc})
            print(f"Train Batch Loss: {loss_val} | Train AUROC: {auroc}")

    epoch_loss = running_loss / dataset_sizes["train"]
    epoch_auroc = running_auroc / dataset_sizes["train"]

    return model, epoch_loss, epoch_auroc, batch_loss


def validate_model(model, dataloader, criterion, optimizer):
    model.eval()

    running_loss = 0.0
    running_auroc = 0.0
    batch_loss = []

    for idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device).flatten()

        optimizer.zero_grad()

        with torch.no_grad():
            outputs = model(inputs).flatten()
            loss = criterion(outputs, labels)

        loss_val = loss.detach()
        batch_loss.append(loss_val)
        running_loss += loss_val * inputs.size(0)

        auroc = binary_auroc(input=outputs, target=labels).item()
        running_auroc += auroc * inputs.size(0)

        if (idx + 1) % 100 == 0:
            wandb.log({"valid_loss": loss_val, "valid_auroc": auroc})
            print(f"valid Batch Loss: {loss_val} | Valid AUROC: {auroc}")

    valid_loss = running_loss / dataset_sizes["val"]
    valid_auroc = running_auroc / dataset_sizes["val"]

    return model, valid_loss, valid_auroc, batch_loss


# Train
run = wandb.init(project="isic_lesions_24", job_type="train")

best_epoch_auroc = -np.inf
best_valid_loss = np.inf
early_stopping_patience = 4
epochs_no_improve = 0

for epoch in range(10):
    model_ft, epoch_loss, train_auroc, train_batch_losses = train_model(
        model_ft, train_dataloader, criterion, optimizer_ft
    )
    model_ft, valid_loss, valid_auroc, valid_batch_losses = validate_model(
        model_ft, valid_dataloader, criterion, optimizer_ft
    )

    print(
        f"Epoch: {epoch} | Train Loss: {epoch_loss} | Train AUROC: {train_auroc} | Valid Loss: {valid_loss} | Valid AUROC: {valid_auroc} \n"
    )
    wandb.log(
        {
            "epoch": epoch,
            "epoch_loss": epoch_loss,
            "epoch_val_loss": valid_loss,
            "epoch_train_auroc": train_auroc,
            "epoch_valid_auroc": valid_auroc,
        }
    )

    # checkpointing dependent on AUROC
    if best_epoch_auroc <= valid_auroc:
        print(f"{b_}Validation AUROC Improved ({best_epoch_auroc} ---> {valid_auroc}){sr_}")
        best_epoch_auroc = valid_auroc
        best_model_wts = copy.deepcopy(model_ft.state_dict())
        PATH = "../models/AUROC{:.4f}_Loss{:.4f}_epoch{:.0f}.bin".format(
            valid_auroc, valid_loss, epoch
        )
        torch.save(model_ft.state_dict(), PATH)
        # Save a model file from the current directory
        print(f"{b_}Model Saved{sr_}")

    # earlystopping dependent on validation loss
    if best_valid_loss >= valid_loss:
        print(f"{b_}Validation Loss Improved ({best_valid_loss} ---> {valid_loss}){sr_}")
        best_valid_loss = valid_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stopping_patience:
        print(
            f"{b_}Early stopping triggered after {epochs_no_improve} epochs with no improvement.{sr_}"
        )
        break

# Load the best model
model_ft.load_state_dict(best_model_wts)


# Compute CV
@torch.inference_mode()
def infer_model(model, dataloader):
    model.eval()

    preds = []
    gts = []

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device).flatten()

        with torch.no_grad():
            outputs = model(inputs).flatten()
            preds.extend(torch.sigmoid(outputs))
            gts.extend(labels)

    preds = [pred.item() for pred in preds]
    gts = [gt.item() for gt in gts]

    return preds, gts


def comp_score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    min_tpr: float = 0.80,
):
    v_gt = abs(np.asarray(solution.values) - 1)
    v_pred = np.array([1.0 - x for x in submission.values])
    max_fpr = abs(1 - min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (
        partial_auc_scaled - 0.5
    )
    return partial_auc


preds, gts = infer_model(model_ft, valid_dataloader)

score = comp_score(
    pd.DataFrame(gts, columns=["target"]),
    pd.DataFrame(preds, columns=["prediction"]),
    "",
)

wandb.log({"pAUC": score})

## Plot pAUC
tpr_threshold = 0.8

fpr, tpr, thresholds = roc_curve(gts, preds)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="blue", lw=2, label="ROC curve (AUC = %0.3f)" % roc_auc)

start_idx = np.where(tpr >= tpr_threshold)[0][0]
fpr_shade = np.concatenate(([fpr[start_idx]], fpr[start_idx:], [1]))
tpr_shade = np.concatenate(([tpr_threshold], tpr[start_idx:], [tpr_threshold]))
plt.fill_between(
    fpr_shade,
    tpr_shade,
    y2=0.8,
    alpha=0.3,
    color="blue",
    label="Partial AUC (TPR >= 0.8) = %0.3f" % score,
)
plt.axhline(y=tpr_threshold, color="red", linestyle="--", label="TPR = 0.80")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")

wandb.log({"pAUC Plot": plt})
