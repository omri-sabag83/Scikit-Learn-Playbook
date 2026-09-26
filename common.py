"""Shared loaders and the locked data split, used by every notebook.

Keeping these in one place means every module sees exactly the same rows:
change the split here and it changes everywhere, never in one notebook only.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path(__file__).resolve().parent / "data" / "raw"
SEED = 2026  # one fixed seed for the split, so it is identical on every run


def load_bank() -> pd.DataFrame:
    """UCI Bank Marketing (bank-full.csv): 45,211 client-campaign rows.

    The target is renamed from the original file's `y` to `subscribed`: 1 if the client subscribed.
    """
    df = pd.read_csv(RAW / "bank-full.csv", sep=";")
    df["subscribed"] = (df.pop("y") == "yes").astype(int)
    return df


def load_credit() -> pd.DataFrame:
    """UCI Default of Credit Card Clients: 30,000 clients, target default = 1 if they defaulted next month.

    The raw file has two header rows (X1..X23 codes, then names); we keep the names.
    September's repayment status is misnamed PAY_0 in the file (there is no PAY_1);
    it's renamed PAY_1 so all three column families number September as 1.
    """
    df = pd.read_csv(RAW / "credit_default.csv", header=1)
    return df.rename(columns={"default payment next month": "default", "PAY_0": "PAY_1"}).set_index("ID")


def credit_split(df: pd.DataFrame | None = None):
    """The locked three-way split for Modules 2-6: 60% train / 20% validation / 20% test.

    Stratified on the target, so each part keeps the ~22% default rate.
    The test part is for Module 6 only. Modules 2-5 must not touch it.
    Returns (train, valid, test) DataFrames.
    """
    if df is None:
        df = load_credit()
    rest, test = train_test_split(df, test_size=0.20, stratify=df["default"], random_state=SEED)
    train, valid = train_test_split(rest, test_size=0.25, stratify=rest["default"], random_state=SEED)
    return train, valid, test


MONTHS = {1: "sep", 2: "aug", 3: "jul", 4: "jun", 5: "may", 6: "apr"}   # PAY_1 = September ... PAY_6 = April


def prepare_credit(d: pd.DataFrame) -> pd.DataFrame:
    """The prepared columns used from Module 3 on (Module 2 builds its own, with only September and August lateness).

    Amounts in thousands of NT$. Months late: the PAY_ status with -2/-1/0 set to 0 (see Module 2, Part 0).
    """
    edu = d["EDUCATION"].map({1: "graduate", 2: "university", 3: "high school"}).fillna("other")
    X = pd.DataFrame({
        "limit_k":        d["LIMIT_BAL"] / 1000,
        "age":            d["AGE"],
        "female":         (d["SEX"] == 2).astype(int),
        "married":        (d["MARRIAGE"] == 1).astype(int),
        "edu_graduate":   (edu == "graduate").astype(int),
        "edu_highschool": (edu == "high school").astype(int),
        "edu_other":      (edu == "other").astype(int),
    }, index=d.index)
    for i, m in MONTHS.items():
        X[f"late_{m}"] = d[f"PAY_{i}"].clip(lower=0)
    X["bill_sep_k"] = d["BILL_AMT1"] / 1000
    X["paid_sep_k"] = d["PAY_AMT1"] / 1000
    return X
