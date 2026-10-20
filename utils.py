

import numpy as np 
import pandas as pd

def sample_uniq_vals_from_series(
    num_uniq_values: int,
    column: pd.Series,
    random_state: int = None
) -> list:
    """
    Sample unique values from a pandas Series.

    Parameters
    ----------
    num_uniq_values : int
        Maximum number of unique values to sample.
    column : pd.Series
        Series from which values are sampled.
    random_state : int or None, default=None
        Seed for reproducible sampling.

    Returns
    -------
    list
        A list containing sampled unique values. At most one missing
        value is included.

    Examples
    --------
    >>> sample_uniq_vals_from_series(
    ...     5,
    ...     xmas_movies["genre"],
    ...     random_state=123
    ... )
    """
    if not isinstance(column, pd.Series):
        raise TypeError("column must be a pandas Series.")

    if not isinstance(num_uniq_values, int):
        raise TypeError("num_uniq_values must be an integer.")

    if num_uniq_values < 0:
        raise ValueError("num_uniq_values cannot be negative.")

    unique_values = column.drop_duplicates()

    sample_size = min(num_uniq_values, len(unique_values))

    sampled_values = unique_values.sample(
        n=sample_size,
        random_state=random_state
    )

    return sampled_values.tolist()


def get_sample_vals_from_each_col( num_uniq_values: int, frame: pd.DataFrame, random_state: int = None) :

    """
    Sample unique values from each col of a pandas DataFrame

    Parameters
    ----------
    num_uniq_values : int
        Maximum number of unique values to sample.
    frame : pd.DataFrame
        DataFrame from which values are sampled.
    random_state : int or None, default=None
        Seed for reproducible sampling.

    Returns
    -------
    None. The function displays sample unique values from each column.

    Notes
    ------
    The function applies sample_uniq_vals_from_series to each column
    in the provided DataFrame, spcified via `frame` parameter.
    """

    for c in frame.columns :
        print(f"Unique values from {c} :\n", '\n', sample_uniq_vals_from_series(num_uniq_values, frame[c], random_state), '\n' )

def get_nans_overview(frame, missing_only=True, decimals=3,ascending=False):
    """
    Summarize missing values in each column of a DataFrame.

    Parameters
    ----------
    frame : pandas.DataFrame
        DataFrame to inspect.
    missing_only : bool, default=True
        If True, include only columns with at least one missing value.
    decimals : int, default=3
        Number of decimal places used to round the results.
    ascending : bool, default=False
        If False, the columns with the most missing values will be displayed on top.

    Returns
    -------
    pandas.DataFrame
        Number and fraction of missing values in each included column,
        sorted by the number of missing values, by default in descending order.
    """
    nans_absolute = frame.isna().sum().sort_values(ascending=ascending)
    nans_relative = nans_absolute.div(frame.shape[0])

    nan_frame = pd.concat(
        [nans_absolute, nans_relative], axis=1
    ).rename(columns={
        0: "Number of NaNs",
        1: "Fraction of NaNs",
    })

    if missing_only:
        nan_frame = nan_frame[nan_frame["Fraction of NaNs"] > 0]

    return nan_frame.round(decimals)