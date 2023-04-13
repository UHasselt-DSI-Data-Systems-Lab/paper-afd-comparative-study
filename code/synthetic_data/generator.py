import copy
import logging
import random
from typing import Any, Dict, List, Optional

import pandas as pd


def assign_fds(settings: Dict[str, Any]) -> Dict[int, int]:
    """## Create FD dictionary
    If a functional dependency tables shall be created, we need to create a dictionary first. This dictionary will represent that true functional dependency assignment between the LHS and the RHS. For each value from the domain of the LHS, a value from the RHS domain will be drawn according to the RHS distribution.
    """

    dictionary = {}
    for left in range(settings["lhs_cardinality"]):
        dictionary[left] = int(
            (settings["rhs_cardinality"]) * settings["rhs_distribution"]()
        )

    return dictionary


def generate_tuples(
    settings: Dict[str, Any], fd_dictionary: Optional[Dict[int, int]]
) -> Dict[int, List[int]]:
    """## Generate tuples
    To generate a tuple, draw a value from the LHS domain according to the LHS distribution. If a FD dictionary is in use, choose the RHS value according to this dictionary. Otherwise, draw a random value from the RHS domain according to the RHS distribution.
    """

    values = {0: [], 1: []}
    for _ in range(settings["tuples"]):
        left = int((settings["lhs_cardinality"]) * settings["lhs_distribution"]())
        values[0].append(left)
        if fd_dictionary:
            values[1].append(fd_dictionary[left])
        else:
            values[1].append(
                int((settings["rhs_cardinality"]) * settings["rhs_distribution"]())
            )
    return values


def get_noise_potential(
    settings: Dict[str, Any], values: Dict[int, List[int]]
) -> float:
    """
    Return the potential percentage of noise that can be introduced to a dataset.
    Values is assumed to be a dictionary with keys 0 and 1, where 0 is the LHS and 1 is the RHS.
    """
    return get_noise_potential_df(pd.DataFrame(values), 0, 1)


def get_noise_potential_df(df: pd.DataFrame, lhs: Any, rhs: Any) -> float:
    """
    Return the potential percentage of noise that can be introduced to a dataset.
    """
    df = df.loc[:, [lhs, rhs]].dropna(how="any")
    counts = df.loc[:, lhs].value_counts()
    potentials = counts.loc[
        (counts // 2) > 0
    ]  # if we change more than half the values, we are decreasing noise again
    return (potentials // 2).sum() / df.shape[0]


def introduce_noise(
    settings: Dict[str, Any], values: Dict[int, List[int]]
) -> Dict[int, List[int]]:
    """
    Introduce noise to a dataset.
    To generate noise, identify all LHS values that occur at least twice. Get the frequency table of those values, take the half of it and sample a list of LHS values (which can occur multiple times) where noise will be introduced into. Identify tuples for each LHS value, and change their RHS value by picking randomly from all other possible RHS values (i.e. from all tuples where the LHS value is not equal to the one of the identified tuple).
    """
    return introduce_noise_copy(settings, values)


def potential_noisy_indices(df: pd.DataFrame, noisy_k: int) -> List[int]:
    """
    Identify all LHS values that bear potential to introduce noise into the dataset df. Returns a list of noisy_k indices from df that can be used to introduce noise. Use this list to iterate through it and change the tuple according to some method for introducing noise.
    """
    X_counts = df.iloc[:, 0].value_counts()  # counts for each generated value from X
    # a list of potential X values to introduce noise to
    # i.e.: X values that appear at least two times
    potentials = X_counts.loc[(X_counts // 2) > 0] // 2
    # from the potentials, sample the X_values that will be changed in the following
    X_values = random.sample(
        potentials.index.tolist(), k=noisy_k, counts=potentials.values.tolist()
    )
    return X_values


def introduce_noise_copy(
    settings: Dict[str, Any], values: Dict[int, List[int]]
) -> Dict[int, List[int]]:
    """
    Introduce noise to a dataset.
    To generate noise, identify all LHS values that occur at least twice. Get the frequency table of those values, take the half of it and sample a list of LHS values (which can occur multiple times) where noise will be introduced into. Identify tuples for each LHS value, and change their RHS value by picking randomly from all other possible RHS values (i.e. from all tuples where the LHS value is not equal to the one of the identified tuple).
    """
    noisy_k = int(settings["noise"] * settings["tuples"])
    if noisy_k == 0:
        return values  # nothing to do
    df = pd.DataFrame(values)
    try:
        X_values = potential_noisy_indices(df, noisy_k)
    except ValueError:
        logging.error(
            f'It is not possible to introduce {settings["noise"]} noise to the dataset. Check the dataset with `get_noise_potential()` first.\n'
        )
        raise ValueError("Cannot create dataset, noise is set too high.")
    dirty_values = copy.deepcopy(values)
    Y_counts = df.iloc[:, 1].value_counts()  # counts for each generated value from Y
    for x, n in pd.Series(X_values).value_counts().items():
        y_candidates = Y_counts[
            Y_counts.index != x
        ].copy()  # get all Y values that are not x
        # get random indices of rows with the X value we want to change
        indices_to_change = df.loc[df.iloc[:, 0] == x].sample(n=n).index
        # an array of Y values to change the identified rows to (i.e. the noise)
        # choosing from existing Y values hopefully maintains the Y distribution
        y_values = random.choices(
            y_candidates.index.tolist(), k=n, weights=y_candidates.values.tolist()
        )
        for i, x_index in enumerate(indices_to_change):
            dirty_values[1][x_index] = y_values[i]
    return dirty_values


def introduce_noise_bogus(
    settings: Dict[str, Any], values: Dict[int, List[int]]
) -> Dict[int, List[int]]:
    """Introduce noise into a clean dataset. Do it in a controlled fashion such that it the noise level set in the settings can be guranteed. The noise will be a completely new value each time, skewing the RHS frequencies heavily. After introducing noise, int(noise * tuples) new RHS values are in the dataset, each one with a frequency of 1."""
    noisy_k = int(settings["noise"] * settings["tuples"])
    if noisy_k == 0:
        return values  # nothing to do
    df = pd.DataFrame(values)
    try:
        X_values = potential_noisy_indices(df, noisy_k)
    except ValueError:
        logging.error(
            f'It is not possible to introduce {settings["noise"]} noise to the dataset. Check the dataset with `get_noise_potential()` first.\n'
        )
        raise ValueError("Cannot create dataset, noise is set too high.")
    dirty_values = copy.deepcopy(values)
    bogus_value = df.iloc[:, 1].nunique()  # starting at nunique is just the best guess
    for x, n in pd.Series(X_values).value_counts().items():
        # get random indices of rows with the X value we want to change
        indices_to_change = df.loc[df.iloc[:, 0] == x].sample(n=n).index
        for i, x_index in enumerate(indices_to_change):
            # make sure that our bogus value is not already in the Y values
            while bogus_value in dirty_values[1]:
                bogus_value += 1
            dirty_values[1][x_index] = bogus_value
    return dirty_values


def introduce_noise_typo(
    settings: Dict[str, Any], values: Dict[int, List[Any]], typos_n: int = 3
) -> Dict[int, List[int]]:
    """Introduce noise into a clean dataset. Will introduce noise in a controlled fashion such that the noise level set in the settings will be guranteed. Noise is introduced by mapping each LHS values to a set of typos_n new RHS values. Those new RHS values represent typos in the data."""
    noisy_k = int(settings["noise"] * settings["tuples"])
    if noisy_k == 0:
        return values  # nothing to do
    df = pd.DataFrame(values)
    try:
        X_values = potential_noisy_indices(df, noisy_k)
    except ValueError:
        logging.error(
            f'It is not possible to introduce {settings["noise"]} noise to the dataset. Check the dataset with `get_noise_potential()` first.\n'
        )
        raise ValueError("Cannot create dataset, noise is set too high.")
    # this mapper will be used to mimick a typo for all values of Y
    typo_mapper = {}
    for v in set(values[1]):
        # the idea is to append a 'random' value to the original one, as if someone tapped an extra key
        type_of_Y = type(v)
        # adding multiple typo options makes it more realistic (I think)
        typo_mapper[v] = [
            type_of_Y(str(v) + str(settings["rhs_cardinality"] + n))
            for n in range(typos_n)
        ]
    dirty_values = copy.deepcopy(values)
    for x, n in pd.Series(X_values).value_counts().items():
        # get random indices of rows with the X value we want to change
        indices_to_change = df.loc[df.iloc[:, 0] == x].sample(n=n).index
        for i, x_index in enumerate(indices_to_change):
            dirty_values[1][x_index] = random.choice(
                typo_mapper[dirty_values[1][x_index]]
            )
    return dirty_values


def generate_SYN(
    fd: bool,
    tuples: int,
    lhs_cardinality: int,
    rhs_cardinality: int,
    lhs_dist_alpha: float,
    lhs_dist_beta: float,
    rhs_dist_alpha: float,
    rhs_dist_beta: float,
    noise: float = 0.01,
    n_type: str = "copy",
) -> pd.DataFrame:
    """The main method to genreate a SYN dataset. Summarizes almost all methods above."""
    settings = {
        "tuples": tuples,
        "lhs_cardinality": lhs_cardinality,
        "rhs_cardinality": rhs_cardinality,
        "lhs_distribution": lambda: random.betavariate(lhs_dist_alpha, lhs_dist_beta),
        "rhs_distribution": lambda: random.betavariate(rhs_dist_alpha, rhs_dist_beta),
        "noise": noise,
    }
    fd_dictionary = None
    if fd:
        fd_dictionary = assign_fds(settings=settings)
    clean_data = generate_tuples(settings=settings, fd_dictionary=fd_dictionary)
    if fd:
        # make sure that noise can be introduced to the clean dataset
        safeguard = 10
        while get_noise_potential(settings, clean_data) < settings["noise"]:
            clean_data = generate_tuples(settings=settings, fd_dictionary=fd_dictionary)
            safeguard -= 1
            if safeguard == 0:
                raise ValueError(
                    f"Could not generate noise for these settings: {settings}"
                )
        if n_type == "copy":
            data = introduce_noise_copy(settings, clean_data)
        elif n_type == "bogus":
            data = introduce_noise_bogus(settings, clean_data)
        elif n_type == "typo":
            data = introduce_noise_typo(settings, clean_data)
    else:
        data = clean_data
    # transform the values to a pandas DataFrame with columns 'lhs','rhs'
    df = pd.DataFrame(data)
    df.columns = ["lhs", "rhs"]
    return df
