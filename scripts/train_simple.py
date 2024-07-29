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
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
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
    torch.set_float32_matmul_precision('highest')

# Set the random seed
set_seed(42)

# Data
def add_path(row):
    return f"../data/train-image/image/{row.isic_id}.jpg"

train_metadata_df = pd.read_csv("../data/stratified_5_fold_train_metadata.csv")
train_metadata_df["path"] = train_metadata_df.apply(lambda row: add_path(row), axis=1)


# dataset
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

# simple resize and normalize
transforms_train = A.Compose([
    A.Resize(124, 124),
    A.Normalize(
        # mean=(0.6962, 0.5209, 0.4193),
        # std=(0.1395, 0.1320, 0.1240)
    ),
    ToTensorV2(),
])

transforms_valid = A.Compose([
    A.Resize(124, 124),
    A.Normalize(
        # mean=(0.6962, 0.5209, 0.4193),
        # std=(0.1395, 0.1320, 0.1240)
    ),
    ToTensorV2(),
])

# dataloaders
# using 2 folds as training and 1 fold as validation
train_df_1 = train_metadata_df.loc[train_metadata_df.fold == 0] # using a subset for training
train_df_2 = train_metadata_df.loc[train_metadata_df.fold == 2] # using a subset for training
train_df_3 = train_metadata_df.loc[train_metadata_df.fold == 3] # using a subset for training
train_df = pd.concat([train_df_1, train_df_2, train_df_3])
valid_df = train_metadata_df.loc[train_metadata_df.fold == 1] # using another fold for validation

num_workers = 24 # based on profiling

train_dataset = SkinDataset(train_df, transform=transforms_train)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, drop_last=True)

valid_dataset = SkinDataset(valid_df, transform=transforms_valid)
valid_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

dataset_sizes = {"train": len(train_dataset), "val": len(valid_dataset)}
print(dataset_sizes)

# calculate bias value
neg_samples = len(train_dataset.get_class_samples(0))
pos_samples = len(train_dataset.get_class_samples(1))
p_positive = pos_samples / (neg_samples + pos_samples)
bias_value = math.log(p_positive / (1 - p_positive))
print(f"Calculated bias value: {bias_value}")


# training and validation utils
def train_model(model, dataloader, criterion, optimizer, train_step, scheduler=None):
    model.train()  # Set model to training mode

    running_loss = 0.0
    running_auroc = 0.0

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

        auroc = binary_auroc(input=outputs, target=labels).item()
        running_auroc += auroc * inputs.size(0)

        if (idx + 1) % 10 == 0:
            train_step += 1
            wandb.log({"train_loss": loss_val, "train_auroc": auroc, "train_step": train_step})
            print(f"Train Batch Loss: {loss_val} | Train AUROC: {auroc}")

    epoch_loss = running_loss / dataset_sizes["train"]
    epoch_auroc = running_auroc / dataset_sizes["train"]

    return model, epoch_loss, epoch_auroc


def validate_model(model, dataloader, criterions, optimizer, valid_step):
    model.eval()

    running_loss = 0.0
    running_auroc = 0.0

    for idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device).flatten().to(torch.float32)

        optimizer.zero_grad()

        with torch.no_grad():
            outputs = model(inputs).flatten()
            loss = criterion(outputs, labels)
        
        loss_val = loss.detach()
        running_loss += loss_val * inputs.size(0)

        auroc = binary_auroc(input=outputs, target=labels).item()
        running_auroc += auroc * inputs.size(0)

        if (idx + 1) % 10 == 0:
            valid_step += 1
            wandb.log({"valid_loss": loss_val, "valid_auroc": auroc, "valid_step": valid_step})
            print(f"valid Batch Loss: {loss_val} | Valid AUROC: {auroc}")

    valid_loss = running_loss / dataset_sizes["val"]
    valid_auroc = running_auroc / dataset_sizes["val"]

    return model, valid_loss, valid_auroc


# model
class SkinClassifier(nn.Module):
    def __init__(self, model_name='resnet18', freeze_backbone=False, bias_value=None):
        super(SkinClassifier, self).__init__()
        
        # Load the specified pre-trained model
        if model_name == 'resnet18':
            self.backbone = models.resnet18(weights="IMAGENET1K_V1")
            if freeze_backbone:
                self.freeze_backbone()
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = self.get_clf_head(num_ftrs, 1, bias_value)
        elif model_name == 'convnext_tiny':
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

    def get_clf_head(self, in_features, out_features, bias_value=None):
        head = nn.Linear(in_features, out_features)
        nn.init.constant_(head.bias, bias_value)
        return head

    def count_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return trainable_params, non_trainable_params

# Create the model
model = SkinClassifier(model_name='efficientnet_v2_s', freeze_backbone=True, bias_value=bias_value)
model = model.to(device)
model = torch.compile(model)
print(model)

trainable_params, non_trainable_params = model.count_parameters()
print(f"Trainable parameters: {trainable_params}")
print(f"Non-trainable parameters: {non_trainable_params}")

# Loss fn and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001,
    weight_decay=1e-6
)

# train loop
run = wandb.init(project="isic_lesions_24", job_type="pretrain")

model_name = "convnext_tiny"
exp_name = "train_linear"

wandb.define_metric("train_step")
wandb.define_metric("valid_step")

train_step = 0
valid_step = 0

best_epoch_auroc = -np.inf
best_valid_loss = np.inf
early_stopping_patience = 4
epochs_no_improve = 0

for epoch in range(15): # reducing epoch to 15 because quick overfitting after correct init
    model_ft, epoch_loss, epoch_train_auroc = train_model(
        model, train_dataloader, criterion, optimizer, train_step,
    )

    model_ft, valid_loss, epoch_valid_auroc = validate_model(
        model, valid_dataloader, criterion, optimizer, valid_step
    )

    wandb.log(
        {
            "epoch": epoch,
            "epoch_loss": epoch_loss,
            "epoch_val_loss": valid_loss,
            "epoch_train_auroc": epoch_train_auroc,
            "epoch_valid_auroc": epoch_valid_auroc
        }
    )

    print(
        f"Epoch: {epoch} | Train Loss: {epoch_loss} | Valid Loss: {valid_loss}\n"
    )
    print(
        f"Epoch: {epoch} | Train AUROC: {epoch_train_auroc} | Valid AUROC: {epoch_valid_auroc}\n"
    )

    # # earlystopping dependent on validation loss
    # if best_valid_loss >= valid_loss:
    #     print(f"{b_}Validation Loss Improved ({best_valid_loss} ---> {valid_loss}){sr_}")
        
    #     # checkpointing
    #     best_model_wts = copy.deepcopy(model_ft.state_dict())
    #     PATH = f"../models/{model_name}_{exp_name}_valid_loss{valid_loss}_epoch{epoch}.bin"
    #     torch.save(model_ft.state_dict(), PATH)
    #     # Save a model file from the current directory
    #     print(f"{b_}Model Saved{sr_}")
    #     best_valid_loss = valid_loss
    #     epochs_no_improve = 0
    # else:
    #     epochs_no_improve += 1

    # if epochs_no_improve >= early_stopping_patience:
    #     print(
    #         f"{b_}Early stopping triggered after {epochs_no_improve} epochs with no improvement.{sr_}"
    #     )
    #     break
