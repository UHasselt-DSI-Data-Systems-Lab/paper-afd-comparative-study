import ctypes
import threading
from typing import Any, List
import re

import numpy as np
import pandas as pd

from . import measures as afd_measures


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


def parallelize_measuring(
    df: pd.DataFrame, table: str, lhs: Any, rhs: Any, measures: List[str]
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
            result[measure] = (
                getattr(afd_measures, measure)(_df, lhs, rhs)
                if not result["trivial_fd"]
                else 1.0
            )
    return result


class TimeoutAfter:
    """A class to allow a controlled timeout."""

    def __init__(self, timeout=(10), exception=TimeoutError):
        self._exception = exception
        self._caller_thread = threading.current_thread()
        self._timeout = timeout
        self._timer = threading.Timer(self._timeout, self.raise_caller)
        self._timer.daemon = True
        self._timer.start()

    def __enter__(self):
        try:
            yield
        finally:
            self._timer.cancel()
        return self

    def __exit__(self, type, value, traceback):
        self._timer.cancel()

    def raise_caller(self):
        ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(self._caller_thread._ident), ctypes.py_object(self._exception)
        )
        if ret == 0:
            raise ValueError("Invalid thread ID")
        elif ret > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(self._caller_thread._ident, None)
            raise SystemError("PyThreadState_SetAsyncExc failed")
