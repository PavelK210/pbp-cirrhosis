"""Общие определения выборки и разбиений для двух ноутбуков."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
CLASSES = ["C", "CL", "D"]
COLORS = {"C": "#216b70", "CL": "#b56c0a", "D": "#a43c58"}
LABS = [
    "Bilirubin", "Cholesterol", "Albumin", "Copper", "Alk_Phos",
    "SGOT", "Tryglicerides", "Platelets", "Prothrombin",
]
LOG_LABS = ["Bilirubin", "Cholesterol", "Copper", "Alk_Phos", "SGOT", "Tryglicerides"]
CATEGORICAL = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage"]
PROFILE = ["Age", *LABS, *CATEGORICAL]


def read_inputs(directory=Path("data")):
    train, test, sample = (
        pd.read_csv(directory / name)
        for name in ("train.csv", "test.csv", "sample_submission.csv")
    )
    expected = {"id", "N_Days", *PROFILE}
    assert set(train.columns) == expected | {"Status"} and set(test.columns) == expected
    assert set(train.Status) == set(CLASSES)
    assert train.id.is_unique and test.id.is_unique and not train.id.isin(test.id).any()
    assert list(sample.columns) == ["id", *(f"Status_{c}" for c in CLASSES)]
    assert test.id.equals(sample.id)
    return train, test, sample


def profile_ids(frame):
    """Одинаковые исходные значения всех 17 признаков получают общий номер."""
    return frame.groupby(PROFILE, sort=False, dropna=False).ngroup().to_numpy()


def checked_splits(frame, seed):
    groups = profile_ids(frame)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(frame[PROFILE], frame.Status, groups))
    for fit, valid in folds:
        assert not np.intersect1d(groups[fit], groups[valid]).size
        assert set(frame.Status.iloc[fit]) == set(frame.Status.iloc[valid]) == set(CLASSES)
    assert np.array_equal(np.sort(np.concatenate([va for _, va in folds])), np.arange(len(frame)))
    return folds


def partition(train):
    """Первый внешний блок — контроль, остальные четыре — выборка разработки."""
    development, control = checked_splits(train, SEED)[0]
    dev = train.iloc[development].reset_index(drop=True)
    inner_folds = checked_splits(dev, 1729)
    return dev, train.iloc[control].reset_index(drop=True), inner_folds
