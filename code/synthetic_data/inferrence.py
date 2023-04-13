from typing import Any, Dict

import pandas as pd


def infer_settings(table: pd.DataFrame, noise: bool = False) -> Dict[str, Any]:
    """Expects table to contain at least two columns. A table will be interpreted as column 0 -> column 1 is a functional dependency if fd = True, otherwise no relation will be assumed.
    From that table, infer tuples, LHS / RHS cardinalities, noise (if fd = True) and the beta distributions for both columns.
    """
    if table.shape[1] < 2:
        raise ValueError(
            "At least two columns in the table are needed for this method."
        )
    table = table.iloc[:, [0, 1]].dropna(how="any", axis="index").copy()
    if table.empty:
        return {"tuples": 0}

    lhs_settings = infer_column_settings(table.iloc[:, 0])
    rhs_settings = infer_column_settings(table.iloc[:, 1])

    found_settings = {
        "tuples": lhs_settings["tuples"],
        "lhs_cardinality": lhs_settings["cardinality"],
        "rhs_cardinality": rhs_settings["cardinality"],
        "lhs_dist_alpha": lhs_settings["dist_alpha"],
        "lhs_dist_beta": lhs_settings["dist_beta"],
        "rhs_dist_alpha": rhs_settings["dist_alpha"],
        "rhs_dist_beta": rhs_settings["dist_beta"],
    }

    # noise only makes sense if we have an FD, otherwise it serves no purpose
    if noise:
        table = table.iloc[:, 0:2]
        table.columns = ["lhs", "rhs"]
        fd_dict_counts = (
            table.groupby(["lhs", "rhs"])
            .count()
            .reset_index(level=1)
            .groupby("lhs")
            .count()
            .copy()
        )
        noises = []
        for lhs, rhs_count in fd_dict_counts.query("rhs > 1").iterrows():
            val_counts = (
                table.query(f'lhs == "{lhs}"')["rhs"]
                .value_counts()
                .sort_values(ascending=False)
            )
            noises.append(val_counts.iloc[1:].sum())
        found_settings["noise"] = sum(noises) / found_settings["tuples"]
    else:
        found_settings["noise"] = 0.0
    return found_settings


def infer_column_settings(column: pd.Series) -> Dict[str, Any]:
    """Infer tuples, cardinality and beta distribution parameters (alpha and beta) of a column supplied as a Pandas Series."""
    import pandas as pd

    found_settings = {}
    found_settings["tuples"] = column.shape[0]
    found_settings["cardinality"] = column.nunique()

    # infer values such that values are in [0, 1] with x_i < x_j if #x_i > #x_j for all i, j in [0, 1, ... #tuples-1]
    val_counts = column.value_counts().sort_values(ascending=False).copy()
    infered_values = pd.Series(
        [
            i / found_settings["cardinality"]
            for i, v in enumerate(val_counts.to_list())
            for _ in range(v)
        ],
        dtype="float64",
    )
    # statistical method of moments inference with two unknown parameters: https://en.wikipedia.org/wiki/Beta_distribution#Two_unknown_parameters
    mean = infered_values.mean()
    variance = infered_values.var()
    if variance == 0:  # without variance, we have a normal distribution
        found_settings["dist_alpha"] = 1.0
        found_settings["dist_beta"] = 1.0
    else:
        found_settings["dist_alpha"] = mean * ((mean * (1 - mean)) / variance - 1)
        found_settings["dist_beta"] = (1 - mean) * ((mean * (1 - mean)) / variance - 1)
    return found_settings
