import itertools
import math
from typing import Any

import numpy as np
import pandas as pd


def rho(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """This measure is rho as proposed to detect soft functional dependencies for CORDS by [Ilyas et al., 2004](https://dl.acm.org/doi/abs/10.1145/1007568.1007641)."""
    xy_counts = df.loc[:, [lhs, rhs]].value_counts().reset_index()
    xy_counts.columns = [lhs, rhs, "xy_count"]
    return 1.0 * df.loc[:, lhs].nunique() / xy_counts.shape[0]


def g1(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """This measure is g_1 as proposed by [Kivinen & Mannila, 1995](https://doi.org/10.1016/0304-3975(95)00028-U)."""
    xy_counts = df.loc[:, [lhs, rhs]].value_counts().reset_index()
    xy_counts.columns = [lhs, rhs, "xy_count"]
    x_counts = df.loc[:, lhs].value_counts().reset_index()
    x_counts.columns = [lhs, "x_count"]
    counts = xy_counts.merge(x_counts, on=lhs)
    violating_tuple_pairs = (
        (counts["xy_count"]) * (counts["x_count"] - counts["xy_count"])
    ).sum()
    return 1.0 - (violating_tuple_pairs / (df.shape[0] ** 2))


def g1_prime(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """This measure is a renormalized version of g_1 proposed by us."""
    xy_counts = df.loc[:, [lhs, rhs]].value_counts().reset_index()
    xy_counts.columns = [lhs, rhs, "xy_count"]
    x_counts = df.loc[:, lhs].value_counts().reset_index()
    x_counts.columns = [lhs, "x_count"]
    counts = xy_counts.merge(x_counts, on=lhs)
    violating_tuple_pairs = (
        (counts["xy_count"]) * (counts["x_count"] - counts["xy_count"])
    ).sum()
    return 1.0 - (
        violating_tuple_pairs
        / ((df.shape[0] ** 2) - (xy_counts["xy_count"].pow(2)).sum())
    )


def shannon_g1(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """This measure calculates 1 - H_R(Y|X), where H_R is the conditional Shannon entropy."""
    r_size = df.shape[0]
    xy_counts = df.loc[:, [lhs, rhs]].value_counts().reset_index()
    xy_counts.columns = [lhs, rhs, "xy_count"]
    x_counts = df.loc[:, lhs].value_counts().reset_index()
    x_counts.columns = [lhs, "x_count"]
    counts = xy_counts.merge(x_counts, on=lhs)
    shannonYX = (
        -1.0
        * (
            (counts["xy_count"] / r_size)
            * np.log2((counts["xy_count"] / r_size) / (counts["x_count"] / r_size))
        ).sum()
    )
    return 1 - shannonYX


def shannon_g1_prime(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """The baselined version of shannon_g1."""
    return max(0, shannon_g1(df, lhs, rhs))


def g2(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """This measure is g_2 as proposed by [Kivinen & Mannila, 1995](https://doi.org/10.1016/0304-3975(95)00028-U)."""
    xy_counts = df.loc[:, [lhs, rhs]].value_counts().reset_index()
    xy_counts.columns = [lhs, rhs, "xy_count"]
    x_counts = df.loc[:, lhs].value_counts().reset_index()
    x_counts.columns = [lhs, "x_count"]
    counts = xy_counts.merge(x_counts, on=lhs)
    violation_participating_tuples = counts[counts["x_count"] > counts["xy_count"]]
    return 1.0 - ((violation_participating_tuples["xy_count"] / df.shape[0]).sum())


def g3(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """This measure is g_3 as proposed by [Kivinen & Mannila, 1995](https://doi.org/10.1016/0304-3975(95)00028-U)."""
    xy_counts = df.loc[:, [lhs, rhs]].value_counts().reset_index()
    xy_counts.columns = [lhs, rhs, "xy_count"]
    x_groups = xy_counts.groupby(lhs)["xy_count"]
    minimum_deletions_needed = (x_groups.sum() - x_groups.max()).sum()
    return 1.0 - (minimum_deletions_needed / df.shape[0])


def g3_prime(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """This measure is g_3 in its renormalized variant proposed by [Giannella & Robertson, 2003](https://doi.org/10.1016/j.is.2003.10.006)."""
    xy_counts = df.loc[:, [lhs, rhs]].value_counts().reset_index()
    xy_counts.columns = [lhs, rhs, "xy_count"]
    x_groups = xy_counts.groupby(lhs)["xy_count"]
    minimum_deletions_needed = (x_groups.sum() - x_groups.max()).sum()
    return 1.0 - (minimum_deletions_needed / (df.shape[0] - df.loc[:, lhs].nunique()))


def fraction_of_information(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """This measures matches the function FD as defined by [Cavallo & Pittarelli, 1987](https://dblp.org/rec/conf/vldb/CavalloP87)."""
    r_size = df.shape[0]
    y_counts = df.loc[:, rhs].value_counts().reset_index()
    y_counts.columns = [rhs, "y_count"]
    shannonY = (
        -1.0
        * ((y_counts["y_count"] / r_size) * np.log2(y_counts["y_count"] / r_size)).sum()
    )
    xy_counts = df.loc[:, [lhs, rhs]].value_counts().reset_index()
    xy_counts.columns = [lhs, rhs, "xy_count"]
    x_counts = df.loc[:, lhs].value_counts().reset_index()
    x_counts.columns = [lhs, "x_count"]
    counts = xy_counts.merge(x_counts, on=lhs)
    shannonYX = (
        -1.0
        * (
            (counts["xy_count"] / r_size)
            * np.log2((counts["xy_count"] / r_size) / (counts["x_count"] / r_size))
        ).sum()
    )
    return (shannonY - shannonYX) / shannonY


def fraction_of_information_prime(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """This measure is a renormalized variant of FI proposed by us, analogous to mu."""
    fi = fraction_of_information(df, lhs, rhs)
    rfi = reliable_fraction_of_information(df, lhs, rhs)
    # note that RFI := FI - E(FI) <=> RFI + FI = -E(FI)
    return rfi / (1 + (rfi - fi))


def smoothed_fraction_of_information(
    df: pd.DataFrame, lhs: Any, rhs: Any, alpha: float = 0.5
) -> float:
    """This measures is smoothed mutual information as defined by [Pennerath, Mandros & Vreeken, 2020](https://doi.org/10.1145/3394486.3403178)."""
    r_size = df.shape[0]
    domX_size = df.loc[:, lhs].nunique()
    domY_size = df.loc[:, rhs].nunique()
    xy_counts = df.loc[:, [lhs, rhs]].value_counts().reset_index()
    xy_counts.columns = [lhs, rhs, "xy_count"]
    x_counts = df.loc[:, lhs].value_counts().reset_index()
    x_counts.columns = [lhs, "x_count"]
    counts = pd.DataFrame(
        itertools.product(df.loc[:, lhs].unique(), df.loc[:, rhs].unique()),
        columns=[lhs, rhs],
    )
    counts = counts.merge(xy_counts, on=[lhs, rhs], how="left")
    counts = counts.merge(x_counts, on=lhs, how="left")
    counts = counts.fillna(0.0)
    smoothedX = (counts["x_count"] + domY_size * alpha) / (
        r_size + domX_size * domY_size * alpha
    )
    smoothedYX = (counts["xy_count"] + alpha) / (r_size + domX_size * domY_size * alpha)
    y_counts = df.loc[:, rhs].value_counts().reset_index()
    y_counts.columns = [rhs, "y_count"]
    smoothedY = (y_counts["y_count"] + domX_size * alpha) / (
        r_size + domX_size * domY_size * alpha
    )
    smoothed_shannonY = -1.0 * (smoothedY * np.log2(smoothedY)).sum()
    smoothed_shannonYX = -1.0 * (smoothedYX * np.log2(smoothedYX / smoothedX)).sum()
    return (smoothed_shannonY - smoothed_shannonYX) / smoothed_shannonY


def reliable_fraction_of_information(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """This measures is reliable fraction of information as defined by [Mandros, Boley & Vreeken, 2017](https://doi.org/10.1145/3097983.3098062)."""
    n = df.shape[0]
    x_counts = df.loc[:, lhs].value_counts()
    y_counts = df.loc[:, rhs].value_counts()
    m = 0.0
    for _, a in x_counts.items():
        comb_n_a = math.comb(n, a)
        for _, b in y_counts.items():
            k0 = max(0, a + b - n)
            p0 = math.comb(b, k0) * math.comb(n - b, a - k0) / comb_n_a
            if k0 != 0:
                m += p0 * (k0 / n) * math.log((k0 * n) / (a * b), 2)

            for k1 in range(k0 + 1, min(a, b) + 1):
                p0 = p0 * (((a - k0) * (b - k0)) / (k1 * (n - a - b + k1)))
                m += p0 * (k1 / n) * math.log((k1 * n) / (a * b), 2)
                k0 = k1
    fi_value = fraction_of_information(df, lhs, rhs)
    bias_estimator = m / (-1.0 * ((y_counts / n) * np.log2(y_counts / n)).sum())
    return fi_value - bias_estimator


def reliable_fraction_of_information_prime(
    df: pd.DataFrame, lhs: Any, rhs: Any
) -> float:
    """This measure is RFI redefined by us to result in a score in the range [0;1]."""
    return max(0.0, reliable_fraction_of_information(df, lhs, rhs))


def pdep_self(df: pd.DataFrame, y: Any) -> float:
    """This measure is pdep(Y) as defined by [Piatetsky-Shapiro & Matheus, 1993](https://www.aaai.org/Library/Workshops/1993/ws93-02-015.php)."""
    return (df.loc[:, y].value_counts() / df.shape[0]).pow(2).sum()


def pdep(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """This measure is pdep as defined by [Piatetsky-Shapiro & Matheus, 1993](https://www.aaai.org/Library/Workshops/1993/ws93-02-015.php)."""
    xy_counts = df.loc[:, [lhs, rhs]].value_counts().reset_index()
    xy_counts.columns = [lhs, rhs, "xy_count"]
    x_counts = df.loc[:, lhs].value_counts().reset_index()
    x_counts.columns = [lhs, "x_count"]
    counts = xy_counts.merge(x_counts, on=lhs)
    return (1 / df.shape[0]) * (counts["xy_count"].pow(2) / counts["x_count"]).sum()


def tau(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """This measure is tau as defined by [Piatetsky-Shapiro & Matheus, 1993](https://www.aaai.org/Library/Workshops/1993/ws93-02-015.php)."""
    pdepXY = pdep(df, lhs, rhs)
    pdepY = pdep_self(df, rhs)
    return (pdepXY - pdepY) / (1 - pdepY)


def mu(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """This measure is mu as defined by [Piatetsky-Shapiro & Matheus, 1993](https://www.aaai.org/Library/Workshops/1993/ws93-02-015.php)."""
    pdepXY = pdep(df, lhs, rhs)
    pdepY = pdep_self(df, rhs)
    r_size = df.shape[0]
    domX_size = df.loc[:, lhs].nunique()
    return 1.0 - ((1 - pdepXY) / (1 - pdepY)) * ((r_size - 1) / (r_size - domX_size))


def mu_prime(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """This measure is mu redefined by us to result in a score in the range [0;1]."""
    return max(0.0, mu(df, lhs, rhs))
