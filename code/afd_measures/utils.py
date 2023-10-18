import multiprocessing as mp
from typing import Any, List
import re

import numpy as np
import pandas as pd

from . import measures as afd_measures

measure_map = {
    "mu_plus": "$\\muplus$",
    "g3_prime": "$g'_3$",
    "g3": "$g_3$",
    "pdep": "$\\pdep$",
    "tau": "$\\tau$",
    "rho": "$\\rho$",
    "g2": "$g_2$",
    "fraction_of_information": "$\\FI$",
    "reliable_fraction_of_information_norm": "$\\RFInorm$",
    "g1_prime": "$g'_1$",
    "g1": "$g_1$",
    "reliable_fraction_of_information_plus": "$\\RFIplus$",
    "smoothed_fraction_of_information": "$\\SFI$",
    "shannon_g1_plus": "$\\SHANNONGONE$",
}

measure_order = [
    "rho",
    "g2",
    "g3",
    "g3_prime",
    "shannon_g1_plus",
    "fraction_of_information",
    "reliable_fraction_of_information_plus",
    "reliable_fraction_of_information_norm",
    "smoothed_fraction_of_information",
    "g1",
    "g1_prime",
    "pdep",
    "tau",
    "mu_plus",
]


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


def add_ground_truth(
    table: str, 
    df: pd.DataFrame,
    gt_path: str = "../../data/ground_truth.csv"
) -> pd.DataFrame:
    gt = pd.read_csv("../../data/ground_truth.csv")
    table = table + ".csv"
    gt = gt.loc[gt.table == table]

    gt_fds = ["{};{}".format(fd["lhs"], fd["rhs"]) for _, fd in gt.iterrows()]
    gt_results = [
        "{};{}".format(fd["lhs"], fd["rhs"]) in gt_fds for _, fd in df.iterrows()
    ]
    df = df.assign(gt=gt_results)

    return df


def parallelize_measuring(
    df: pd.DataFrame,
    table: str,
    lhs: Any,
    rhs: Any,
    measures: List[str],
    timeout: int = 30,
):
    """Common method to parallelize measuring df[lhs] -> df[rhs] of a table namen table."""
    result = {
        "table": table,
        "lhs": lhs,
        "rhs": rhs,
    }
    _df = df.loc[:, [lhs, rhs]].dropna().copy()
    if _df.empty:
        result["empty"] = True
        return result
    result["trivial_fd"] = is_trivial_fd(_df, lhs, rhs)
    result["exact_fd"] = is_perfect_fd(_df, lhs, rhs)
    for measure in measures:
        if result["trivial_fd"]:
            result[measure] = 1.0
        else:
            result[measure] = run_with_timeout(
                getattr(afd_measures, measure),
                args=(_df, lhs, rhs),
                default=np.NaN,
                timeout=timeout,
            )
    return result


def run_with_timeout(func, args=(), kwds={}, timeout=30, default=None):
    """A method that uses multiprocessing to timeout a function call. Taken from stackoverflow, thanks to the user `unutbu` for sharing it: https://stackoverflow.com/a/13822315 ."""
    pool = mp.Pool(processes=1)
    result = pool.apply_async(func, args=args, kwds=kwds)
    try:
        val = result.get(timeout=timeout)
    except mp.TimeoutError:
        pool.terminate()
        return default
    else:
        pool.close()
        pool.join()
        return val
