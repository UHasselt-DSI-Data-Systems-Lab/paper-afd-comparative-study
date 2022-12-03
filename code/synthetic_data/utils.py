import math

import numpy as np
import pandas as pd


def create_skew_lookup(
    alpha_precision: float = 0.01, beta_precision: float = 0.1
) -> pd.DataFrame:
    """Create a simple lookup table for possible alpha, beta and their repsective skew values for a given precision."""
    alpha = np.arange(0.0 + alpha_precision, 1.0 + alpha_precision, alpha_precision)
    beta = np.arange(1.0, 10.0 + beta_precision, beta_precision)
    return pd.DataFrame(
        {"skew": {(a, b): beta_skewness(a, b) for a in alpha for b in beta}}
    )


def beta_skewness(alpha: float, beta: float) -> float:
    """Return the skewness of the Beta distribution given the alpha and beta values."""
    return (2 * (beta - alpha) * math.sqrt(alpha + beta + 1)) / (
        (alpha + beta + 2) * math.sqrt(alpha * beta)
    )
