import os

print(os.cpu_count())

import gc
import re
import math
from glob import glob
import wandb

wandb.require("core")

import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

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

run = wandb.init(
    project="isic_lesions_24",
    job_type="evaluate_folds",
)

train_metadata_df = pd.read_csv("../data/stratified_4_fold_new.csv")


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


transforms_valid = A.Compose(
    [
        A.Resize(384, 384),
        A.Normalize(),
        ToTensorV2(),
    ]
)


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


def get_dataloaders_and_stats(fold):
    valid_df = train_metadata_df.loc[train_metadata_df.fold == fold]

    num_workers = 24  # based on profiling

    file_hdf = "/home/ubuntu/ayusht/skin/data/train-image.hdf5"
    valid_dataset = SkinDataset(valid_df, file_hdf, transform=transforms_valid)
    dataset_sizes = {"val": len(valid_dataset)}
    print(dataset_sizes)

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    return valid_dataloader


model_files = glob("../models/*.bin")

def filter_files_by_run_id(files, run_id):
    """
    Filters the list of files by the given run ID.

    Args:
    files (list of str): List of file paths.
    run_id (str): The run ID to filter by.

    Returns:
    list of str: List of file paths that contain the specified run ID.
    """
    pattern = re.compile(rf"_{run_id}_")
    filtered_files = [file for file in files if pattern.search(file)]
    return filtered_files


def extract_valid_loss(file_name):
    """
    Extracts the valid_loss value from the given file name.

    Args:
    file_name (str): The file name from which to extract the valid_loss.

    Returns:
    float: The extracted valid_loss value.
    """
    match = re.search(r'valid_loss([\d\.]+)', file_name)
    if match:
        return float(match.group(1))
    return float('inf')  # Return a very high value if valid_loss is not found

def sort_files_by_valid_loss(files):
    """
    Sorts the list of files by the valid_loss value in the file names.

    Args:
    files (list of str): List of file paths to sort.

    Returns:
    list of str: Sorted list of file paths by valid_loss.
    """
    return sorted(files, key=extract_valid_loss)


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


def log_positive_dist(preds):
    positive_preds = np.array(preds)[np.array(gts) == 1.0]

    # Plot the distribution
    plt.figure(figsize=(8, 6))
    plt.hist(positive_preds, bins=100)
    plt.title('Prediction Distribution for Positive Class')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')

    return plt


def log_negative_dist(preds):
    negative_preds = np.array(preds)[np.array(gts) == 0.0]

    # Plot the distribution
    plt.figure(figsize=(8, 6))
    plt.hist(negative_preds, bins=100)
    plt.title('Prediction Distribution for Negative Class')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')

    return plt


run_ids = [
    "d9e6sc58", # fold 0
    "7gpz5owu", # fold 1
    "ylb6uvtj", # fold 2
    "aa7biks12", # fold 3
]

scores = {}

for fold, run_id in enumerate(run_ids):
    if run_id not in scores.keys():
        scores[run_id] = {}
    # dataloader
    valid_dataloader = get_dataloaders_and_stats(fold)

    # models
    runid_model_files = filter_files_by_run_id(model_files, run_id)
    selected_model_files = sort_files_by_valid_loss(runid_model_files)[:3]
    
    for path in selected_model_files:
        scores[run_id][path] = {}

        # run = wandb.init(
        #     project="isic_lesions_24",
        #     job_type="evaluate_folds",
        #     name=f'{run_id}_fold_{fold}_{path.split("/")[-1]}_eval'
        # )

        model = SkinClassifier(
            model_name="efficientnet_v2_s"
        )
        model = model.to(device)
        model = torch.compile(model)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)

        preds, gts = infer_model(model, valid_dataloader)
        score = comp_score(
            pd.DataFrame(gts, columns=["target"]),
            pd.DataFrame(preds, columns=["prediction"]),
            ""
        )
        auroc = roc_auc_score(gts, preds)

        # pos_plt = log_positive_dist(preds)
        # neg_plt = log_negative_dist(preds)

        scores[run_id][path]["pAUC"] = score
        scores[run_id][path]["AUROC"] = auroc

        # wandb.log({
        #     "pAUC": score,
        #     "AUROC": auroc,
        # })

        # wandb.finish()

        model.to("cpu")
        del model
        gc.collect()

    del valid_dataloader
    gc.collect()


ensembles_df = pd.DataFrame(
    columns=[
        "fold0", "fold1", "fold2", "fold3", "val_loss0", "val_loss1", "val_loss2", "val_loss3", "pAUC", "std_pAUC", "AUROC", "atd_AUROC"
    ]
)

ensemble_scores = {}

fold_0_scores = scores[run_ids[0]]
fold_1_scores = scores[run_ids[1]]
fold_2_scores = scores[run_ids[2]]
fold_3_scores = scores[run_ids[3]]


from itertools import product


# Create a list to store results
results = []

# Calculate the scores for every combination of models
for model_0, model_1, model_2, model_3 in product(fold_0_scores.keys(), fold_1_scores.keys(), fold_2_scores.keys(), fold_3_scores.keys()):
    combined_pAUC = np.mean([fold_0_scores[model_0]['pAUC'], fold_1_scores[model_1]['pAUC'], fold_2_scores[model_2]['pAUC'], fold_3_scores[model_3]['pAUC']])
    combined_AUROC = np.mean([fold_0_scores[model_0]['AUROC'], fold_1_scores[model_1]['AUROC'], fold_2_scores[model_2]['AUROC'], fold_3_scores[model_3]['AUROC']])
    
    # Calculate std deviation
    std_pAUC = np.std([fold_0_scores[model_0]['pAUC'], fold_1_scores[model_1]['pAUC'], fold_2_scores[model_2]['pAUC'], fold_3_scores[model_3]['pAUC']])
    std_AUROC = np.std([fold_0_scores[model_0]['AUROC'], fold_1_scores[model_1]['AUROC'], fold_2_scores[model_2]['AUROC'], fold_3_scores[model_3]['AUROC']])
    
    # Extract val losses from model names
    val_loss0 = float(model_0.split('_')[5][4:])
    val_loss1 = float(model_1.split('_')[5][4:])
    val_loss2 = float(model_2.split('_')[5][4:])
    val_loss3 = float(model_3.split('_')[5][4:])
    
    results.append([model_0, model_1, model_2, model_3, val_loss0, val_loss1, val_loss2, val_loss3, combined_pAUC, std_pAUC, combined_AUROC, std_AUROC])

# Create a DataFrame
ensembles_df = pd.DataFrame(
    results,
    columns=["fold0", "fold1", "fold2", "fold3", "val_loss0", "val_loss1", "val_loss2", "val_loss3", "pAUC", "std_pAUC", "AUROC", "std_AUROC"]
)

wandb.log({"ensembles_4_fold_combinations": ensembles_df})
ensembles_df.to_csv("../data/ensembles_4_fold_combinations.csv")
