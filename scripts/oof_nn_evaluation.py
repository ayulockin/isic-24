import os

print(os.cpu_count())

import copy
import math
import wandb

wandb.require("core")

import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
import torch.optim as optim
from torch import nn
from torchvision import models
import torch.backends.cudnn as cudnn

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torcheval.metrics.functional import binary_auroc

from sklearn.metrics import roc_auc_score
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

num_workers = 24


# Data
def add_path(row):
    return f"../data/train-image/image/{row.isic_id}.jpg"

train_metadata_df = pd.read_csv("../data/stratified_5_fold_train_metadata.csv")
train_metadata_df["path"] = train_metadata_df.apply(lambda row: add_path(row), axis=1)
train_metadata_df = train_metadata_df[["path", "target", "fold"]]


# dataset
class SkinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, target_transform=None):
        assert "path" in df.columns
        assert "target" in df.columns

        self.paths = df.path.tolist()
        self.labels = df.target.tolist()  # binary
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = read_image(self.paths[idx]).to(torch.uint8)
        label = self.labels[idx] / 1.0
        if self.transform:
            image = image.numpy().transpose((1, 2, 0))
            image = self.transform(image=image)["image"]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_class_samples(self, class_label):
        indices = [i for i, label in enumerate(self.labels) if label == class_label]
        return indices


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

        # unfreeze last conv block
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

transforms_valid = A.Compose(
    [
        A.Resize(124, 124),
        A.Normalize(),
        ToTensorV2(),
    ]
)

valid_df_fold0 = train_metadata_df.loc[train_metadata_df.fold == 1]
valid_df_fold1 = train_metadata_df.loc[train_metadata_df.fold == 0]
valid_df_fold2 = train_metadata_df.loc[train_metadata_df.fold == 2]
valid_df_fold3 = train_metadata_df.loc[train_metadata_df.fold == 3]
valid_df_fold4 = train_metadata_df.loc[train_metadata_df.fold == 4]

oof_dfs = [
    valid_df_fold0, valid_df_fold1, valid_df_fold2, valid_df_fold3, valid_df_fold4
]

model_paths = [
    "/home/ubuntu/ayusht/skin/models/efficientnet_v2_s_enxwbyys_valid_loss5.525981903076172_epoch25.bin",
    "/home/ubuntu/ayusht/skin/models/efficientnet_v2_s_lmfhy5jn_valid_loss20807.064453125_epoch2.bin",
    "/home/ubuntu/ayusht/skin/models/efficientnet_v2_s_r48h8o37_valid_loss4399.478515625_epoch0.bin",
    "/home/ubuntu/ayusht/skin/models/efficientnet_v2_s_ladxeeek_valid_loss5.642144203186035_epoch14.bin",
    "/home/ubuntu/ayusht/skin/models/efficientnet_v2_s_qkjqdyt7_valid_loss10.56924819946289_epoch10.bin",
]


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


def comp_score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float=0.80):
    v_gt = abs(np.asarray(solution.values)-1)
    v_pred = np.array([1.0 - x for x in submission.values])
    max_fpr = abs(1-min_tpr)
    partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
    # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
    # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
    partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    return partial_auc

gts = []
preds = []

for valid_df, model_path in zip(oof_dfs, model_paths):
    valid_dataset = SkinDataset(valid_df, transform=transforms_valid)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model = SkinClassifier(
        model_name="efficientnet_v2_s", freeze_backbone=True,
    )
    model = model.to(device)
    model = torch.compile(model)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    _preds, _gts = infer_model(model, valid_dataloader)

    gts.extend(_gts)
    preds.extend(_preds)

pAUC = comp_score(
    pd.DataFrame(gts, columns=["target"]),
    pd.DataFrame(preds, columns=["prediction"]),
    ""
)
print("pAUC: ", pAUC)

# few models are bad - remove early stopping and train