import re

import pandas as pd


def clean_colname(col: str) -> str:
    """Cleans the column names."""
    return (
        re.sub("[^(a-z)(A-Z)(0-9)._-]", "", col)
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
    )


def is_perfect_fd(df: pd.DataFrame, lhs: str, rhs: str) -> bool:
    """Check if lhs -> rhs in df is a perfect functional dependency."""
    unique = df.drop_duplicates(subset=[lhs, rhs], ignore_index=True).copy()
    return unique.loc[:, lhs].nunique() == unique.shape[0]


def is_trivial_fd(df: pd.DataFrame, lhs: str, rhs: str) -> bool:
    """Check if lhs -> rhs in df is a trivial functional dependency, i.e. LHS is a key or RHS is only one value."""
    _df = df.dropna(subset=[lhs, rhs]).copy()
    return _df.loc[:, lhs].nunique() == _df.shape[0] or _df.loc[:, rhs].nunique() == 1

def add_ground_truth(table: str, df: pd.DataFrame):
    gt = pd.read_csv("../../data/ground_truth.csv")
    table = table + ".csv"
    gt = gt.loc[gt.table == table]

    gt_fds = ["{};{}".format(fd["lhs"], fd["rhs"]) for _,fd in gt.iterrows()]
    gt_results = ["{};{}".format(fd["lhs"], fd["rhs"]) in gt_fds for _,fd in df.iterrows()]
    df = df.assign(gt=gt_results)

    return df