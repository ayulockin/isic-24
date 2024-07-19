import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import wandb
wandb.require("core")
import lightgbm as lgb

import optuna

df_train = pd.read_csv("../data/stratified_5_fold_train_metadata.csv")


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


num_cols = [
    "age_approx",
    "clin_size_long_diam_mm",
    "tbp_lv_A",
    "tbp_lv_Aext",
    "tbp_lv_B",
    "tbp_lv_Bext",
    "tbp_lv_C",
    "tbp_lv_Cext",
    "tbp_lv_H",
    "tbp_lv_Hext",
    "tbp_lv_L",
    "tbp_lv_Lext",
    "tbp_lv_areaMM2",
    "tbp_lv_area_perim_ratio",
    "tbp_lv_color_std_mean",
    "tbp_lv_deltaA",
    "tbp_lv_deltaB",
    "tbp_lv_deltaL",
    "tbp_lv_deltaLB",
    "tbp_lv_deltaLBnorm",
    "tbp_lv_eccentricity",
    "tbp_lv_minorAxisMM",
    "tbp_lv_nevi_confidence",
    "tbp_lv_norm_border",
    "tbp_lv_norm_color",
    "tbp_lv_perimeterMM",
    "tbp_lv_radial_color_std_max",
    "tbp_lv_stdL",
    "tbp_lv_stdLExt",
    "tbp_lv_symm_2axis",
    "tbp_lv_symm_2axis_angle",
    "tbp_lv_x",
    "tbp_lv_y",
    "tbp_lv_z",
]
new_num_cols = [
    "lesion_size_ratio",
    "lesion_shape_index",
    "hue_contrast",
    "luminance_contrast",
    "lesion_color_difference",
    "border_complexity",
    "color_uniformity",
    "3d_position_distance",
    "perimeter_to_area_ratio",
    "lesion_visibility_score",
    "symmetry_border_consistency",
    "color_consistency",
    "size_age_interaction",
    "hue_color_std_interaction",
    "lesion_severity_index",
    "shape_complexity_index",
    "color_contrast_index",
    "log_lesion_area",
    "normalized_lesion_size",
    "mean_hue_difference",
    "std_dev_contrast",
    "color_shape_composite_index",
    "3d_lesion_orientation",
    "overall_color_difference",
    "symmetry_perimeter_interaction",
    "comprehensive_lesion_index",
    "color_variance_ratio",
    "border_color_interaction",
    "size_color_contrast_ratio",
    "age_normalized_nevi_confidence",
    "color_asymmetry_index",
    "3d_volume_approximation",
    "color_range",
    "shape_color_consistency",
    "border_length_ratio",
    "age_size_symmetry_index",
]
new_cat_cols = ["combined_anatomical_site"]
cat_cols = ["sex", "tbp_tile_type", "tbp_lv_location", "tbp_lv_location_simple"]
train_cols = num_cols + new_num_cols + cat_cols + new_cat_cols


def objective(trial):
    print(trial._trial_id)
    run = wandb.init(project="isic_lesions_24", group="lgbm_train_hp_0")

    lgb_params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "device": "gpu",
        "verbosity": -1,
    }

    run.config.update(lgb_params)

    scores = []
    for fold in range(5):
        _df_train = df_train[df_train["fold"] != fold].reset_index(drop=True)
        _df_valid = df_train[df_train["fold"] == fold].reset_index(drop=True)
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(_df_train[train_cols], _df_train["target"])
        preds = model.predict(_df_valid[train_cols])
        score = comp_score(
            _df_valid[["target"]], pd.DataFrame(preds, columns=["prediction"]), ""
        )
        print(f"fold: {fold} - Partial AUC Score: {score:.5f}")
        scores.append(score)

    score = np.mean(scores)
    print(f"LGBM Score: {score:.5f}")
    run.log({"pAUC": score})
    run.finish()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)
 
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
