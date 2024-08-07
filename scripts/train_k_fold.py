import os

print(os.cpu_count())

import gc
import math
import wandb

wandb.require("core")

import random
import pandas as pd
import numpy as np
from tqdm import tqdm

import h5py
from PIL import Image
from io import BytesIO

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
from torch import nn
from torchvision import models

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torcheval.metrics.functional import binary_auroc

from sklearn.model_selection import StratifiedGroupKFold

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


# Set the random seed
set_seed(42)

# Data
train_metadata_df = pd.read_csv("../data/train-metadata.csv")
print(f"Train: {len(train_metadata_df)}")

N_SPLITS = 4

gkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

train_metadata_df["fold"] = -1
for idx, (train_idx, val_idx) in enumerate(
    gkf.split(
        train_metadata_df,
        train_metadata_df["target"],
        groups=train_metadata_df["patient_id"],
    )
):
    train_metadata_df.loc[val_idx, "fold"] = idx

train_metadata_df = train_metadata_df[["isic_id", "target", "fold"]]
# TODO: random name so that it doesn't overwrite
train_metadata_df.to_csv("../data/stratified_4_fold.csv", index=False)


# dataset
class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, file_hdf: str, transform=None):
        assert "isic_id" in df.columns
        assert "target" in df.columns

        self.fp_hdf = h5py.File(file_hdf, mode="r")
        self.isic_ids = df["isic_id"].values
        self.labels = df.target.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, idx: int):
        isic_id = self.isic_ids[idx]
        image = np.array(Image.open(BytesIO(self.fp_hdf[isic_id][()])))
        label = self.labels[idx] / 1.0
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label

    def get_class_samples(self, class_label):
        indices = [i for i, label in enumerate(self.labels) if label == class_label]
        return indices


# simple resize and normalize
transforms_train = A.Compose(
    [
        A.Resize(224, 224),
        A.CenterCrop(
            height=224,
            width=224,
            p=1.0,
        ),
        A.CLAHE(clip_limit=4, tile_grid_size=(10, 10), p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=90, p=0.6),
        A.HueSaturationValue(
            hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.RandomRotate90(p=0.6),
        A.Flip(p=0.7),
        A.Normalize(),
        ToTensorV2(),
    ]
)

transforms_valid = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2(),
    ]
)


def get_dataloaders_and_stats(fold):
    # dataloaders
    train_df = train_metadata_df.loc[train_metadata_df.fold != fold]
    valid_df = train_metadata_df.loc[train_metadata_df.fold == fold]

    num_workers = 24  # based on profiling

    file_hdf = "/home/ubuntu/ayusht/skin/data/train-image.hdf5"
    train_dataset = SkinDataset(train_df, file_hdf, transform=transforms_train)
    valid_dataset = SkinDataset(valid_df, file_hdf, transform=transforms_valid)
    dataset_sizes = {"train": len(train_dataset), "val": len(valid_dataset)}
    print(dataset_sizes)

    # calculate bias value
    neg_samples = len(train_dataset.get_class_samples(0))
    pos_samples = len(train_dataset.get_class_samples(1))
    p_positive = pos_samples / (neg_samples + pos_samples)
    bias_value = math.log(p_positive / (1 - p_positive))
    print(f"Calculated bias value: {bias_value}")

    # calculate class weight
    pos_weight = torch.ones([1]) * (neg_samples / pos_samples)
    pos_weight = pos_weight.to(device)
    print(f"Calculated pos_weight: {pos_weight}")

    # calculate weight for each class for random sampler
    neg_wts = 1 / neg_samples
    pos_wts = 1 / pos_samples
    sample_wts = []

    for label in train_dataset.labels:
        if label == 0:
            sample_wts.append(neg_wts)
        else:
            sample_wts.append(pos_wts)

    sampler = WeightedRandomSampler(
        weights=sample_wts, num_samples=int(len(train_dataset) / 3), replacement=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=128,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return (
        train_dataloader,
        valid_dataloader,
        dataset_sizes,
        bias_value,
        pos_weight,
    )


# training and validation utils
def train_model(
    model,
    dataloader,
    criterion,
    optimizer,
    train_step,
    dataset_sizes,
    scheduler=None,
    debug=False,
):
    model.train()

    running_loss = 0.0
    preds = []
    gts = []

    for idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device).flatten().to(torch.float32)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs).flatten()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        loss_val = loss.detach()
        running_loss += loss_val * inputs.size(0)

        preds.extend(outputs)
        gts.extend(labels)

        if (idx + 1) % 10 == 0:
            train_step += 1
            wandb.log(
                {
                    "train_loss": loss_val,
                    "train_step": train_step,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            print(f"Train Batch Loss: {loss_val}")

        if scheduler:
            scheduler.step()

        if debug:
            break

    epoch_loss = running_loss / dataset_sizes["train"]
    epoch_auroc = binary_auroc(
        input=torch.tensor(preds).to(device), target=torch.tensor(gts).to(device)
    ).item()

    return model, epoch_loss, epoch_auroc


def validate_model(
    model, dataloader, criterion, optimizer, valid_step, dataset_sizes, debug=False
):
    model.eval()

    running_loss = 0.0
    preds = []
    gts = []

    for idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device).flatten().to(torch.float32)

        optimizer.zero_grad()

        with torch.no_grad():
            outputs = model(inputs).flatten()
            loss = criterion(outputs, labels)

        loss_val = loss.detach()
        running_loss += loss_val * inputs.size(0)

        preds.extend(outputs)
        gts.extend(labels)

        if (idx + 1) % 10 == 0:
            valid_step += 1
            wandb.log({"valid_loss": loss_val, "valid_step": valid_step})
            print(f"valid Batch Loss: {loss_val}")

        if debug:
            break

    valid_loss = running_loss / dataset_sizes["val"]
    valid_auroc = binary_auroc(
        input=torch.tensor(preds).to(device), target=torch.tensor(gts).to(device)
    ).item()

    return model, valid_loss, valid_auroc


# model
class SkinClassifier(nn.Module):
    def __init__(self, model_name="resnet18", freeze_backbone=False, bias_value=None):
        super(SkinClassifier, self).__init__()

        # Load the specified pre-trained model
        if model_name == "resnet18":
            self.backbone = models.resnet18(weights="IMAGENET1K_V1")
            if freeze_backbone:
                self.freeze_backbone()
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = self.get_clf_head(num_ftrs, 1, bias_value)
        elif model_name == "convnext_tiny":
            self.backbone = models.convnext_tiny(weights="IMAGENET1K_V1")
            if freeze_backbone:
                self.freeze_backbone()
            num_ftrs = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = self.get_clf_head(num_ftrs, 1, bias_value)
        elif model_name == "efficientnet_v2_s":
            self.backbone = models.efficientnet_v2_s(weights="IMAGENET1K_V1")
            if freeze_backbone:
                self.freeze_backbone()
            num_ftrs = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = self.get_clf_head(num_ftrs, 1, bias_value)
        elif model_name == "efficientnet_v2_m":
            self.backbone = models.efficientnet_v2_m(weights="IMAGENET1K_V1")
            if freeze_backbone:
                self.freeze_backbone()
            num_ftrs = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = self.get_clf_head(num_ftrs, 1, bias_value)
        elif model_name == "mobilenet_v3_small":
            self.backbone = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
            if freeze_backbone:
                self.freeze_backbone()
            num_ftrs = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = self.get_clf_head(num_ftrs, 1, bias_value)
        else:
            raise ValueError(f"Model {model_name} not supported")

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.features[6].parameters():
            param.requires_grad = True

        for param in self.backbone.features[7].parameters():
            param.requires_grad = True

    def get_clf_head(self, in_features, out_features, bias_value=None):
        head = nn.Linear(in_features, out_features)
        if bias_value:
            nn.init.constant_(head.bias, bias_value)
        return head

    def count_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )
        return trainable_params, non_trainable_params


def train_and_validate_folds(fold):
    # Initialize wandb
    run = wandb.init(
        project="isic_lesions_24", job_type="4_fold_nn", name=f"fold_{fold}"
    )
    wandb.define_metric("train_step")
    wandb.define_metric("valid_step")

    model_name = "efficientnet_v2_s"
    debug = False
    epochs = 62 if not debug else 1

    # Get data and stats
    train_dataloader, valid_dataloader, dataset_sizes, bias_value, pos_weight = (
        get_dataloaders_and_stats(fold)
    )

    # Create the model
    model = SkinClassifier(
        model_name=model_name, freeze_backbone=True, bias_value=bias_value
    )
    model = model.to(device)
    model = torch.compile(model)
    print(model)

    trainable_params, non_trainable_params = model.count_parameters()
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

    # Loss fn and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=781 * 2, T_mult=2, eta_min=1e-6, last_epoch=-1
    )

    # train loop
    train_step = 0
    valid_step = 0
    best_epoch_auroc = -np.inf
    best_valid_loss = np.inf
    early_stopping_patience = 4
    epochs_no_improve = 0

    for epoch in range(
        epochs
    ):  # reducing epoch to 15 because quick overfitting after correct init
        model, epoch_loss, epoch_train_auroc = train_model(
            model,
            train_dataloader,
            criterion,
            optimizer,
            train_step,
            dataset_sizes,
            scheduler,
            debug=debug,
        )

        model, valid_loss, epoch_valid_auroc = validate_model(
            model,
            valid_dataloader,
            criterion,
            optimizer,
            valid_step,
            dataset_sizes,
            debug=debug,
        )

        wandb.log(
            {
                "epoch": epoch,
                "epoch_loss": epoch_loss,
                "epoch_val_loss": valid_loss,
                "epoch_train_auroc": epoch_train_auroc,
                "epoch_valid_auroc": epoch_valid_auroc,
            }
        )

        print(f"Epoch: {epoch} | Train Loss: {epoch_loss} | Valid Loss: {valid_loss}\n")
        print(
            f"Epoch: {epoch} | Train AUROC: {epoch_train_auroc} | Valid AUROC: {epoch_valid_auroc}\n"
        )

        # earlystopping dependent on validation loss
        if best_valid_loss >= valid_loss:
            print(
                f"{b_}Validation Loss Improved ({best_valid_loss} ---> {valid_loss}){sr_}"
            )

            # checkpointing
            PATH = f"../models/{model_name}_{run.id}_valid_loss{valid_loss}_epoch{epoch}_fold{fold}.bin"
            torch.save(model.state_dict(), PATH)
            print(f"{b_}Model Saved{sr_}")
            best_valid_loss = valid_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

    # end training for this fold
    # offload to cpu
    model = model.to("cpu")
    del model
    del train_dataloader
    del valid_dataloader
    gc.collect()

    # finish wandb run
    run.finish()


if __name__ == "__main__":
    for fold in range(N_SPLITS):
        print(f"Fold: {fold}")
        train_and_validate_folds(fold)
