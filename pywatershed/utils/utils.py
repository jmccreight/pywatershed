import functools
import pathlib as pl
from time import time
from typing import Optional

import numpy as np
import pandas as pd


def timer(func):
    """Use as a decorator to print the execution time of the passed function"""

    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2 - t1):.4f}s")
        return result

    return wrap_func


def diff_dicts(dict_a: dict, dict_b: dict, ignore_keys: list = []):
    """Diff two dictionaries

    Args:
        dict_a: the first dictionary
        dict_b: the second dictionary
    """
    keys_a = dict_a.keys()
    keys_b = dict_b.keys()
    if keys_a_not_b := keys_a - keys_b:
        print(f"keys in a but not b: {keys_a_not_b}")
        print("")

    if keys_b_not_a := keys_b - keys_a:
        print(f"keys in b but not a: {keys_b_not_a}")
        print("")

    keys_both = keys_a & keys_b
    for kk in keys_both:
        if kk in ignore_keys:
            continue
        val_a = dict_a[kk]
        val_b = dict_b[kk]
        try:
            np.testing.assert_equal(val_a, val_b)
        except AssertionError:
            print(f'key "{kk}" does not match')
            print("value for a: ")
            print(f"    {val_a}")
            print("value for b: ")
            print(f"    {val_b}")
            print("")


def pyprms_control_no_defaults(
    control_file: pl.Path,
    metadata,
    verbose: Optional[bool] = False,
):
    """Get a pyPRMS Control object where no defaults are applied.

    Only necessary until pypRMS PR #40 is merged.
    """
    import pyPRMS as pp

    pp_control = pp.ControlFile(
        filename=control_file, metadata=metadata, verbose=verbose
    )
    return pp_control


def write_data_file(df: pd.DataFrame, output_file_path: pl.Path) -> None:
    """pyPRMS does not have this capability to write PRMS data files.

    Currently only implemented for data_files containing only runoff obs, if
    other variables are present a NotImplementedError will be raised.

    Args:
        df: pd.DataFrame obtained from pyPRMS via DataFile(file).data and
          potentially subset in columns or rows.
        output_file_path: The path where to write the new data file.
    """
    df = df.copy().fillna(value=-999.0)
    runoff_mask = df.columns.str.contains("runoff")
    if not runoff_mask.all():
        raise NotImplementedError("")
    # >
    hash_57 = "#" * 57
    slash_73 = "/" * 73
    slash_2 = "/" * 2
    lb = "\n"

    stn_ids = df.columns.str.slice(7).tolist()

    with open(output_file_path, "w") as file:
        file.write("Created by pywatershed" + lb)
        file.write(slash_73 + lb)
        file.write(slash_2 + " Station IDs for runoff" + lb)
        file.write(slash_2 + " ID" + lb)
        for ss in stn_ids:
            file.write(slash_2 + " " + ss + lb)
        # <
        file.write(slash_73 + lb)
        file.write(slash_2 + " Unit: runoff = cfs" + lb)
        file.write(slash_73 + lb)
        file.write("runoff " + f"{len(stn_ids)}" + lb)
        file.write(hash_57 + lb)
        for index, row in df.iterrows():
            time = index.strftime("%Y %m %d 0 0 0")
            data = ""
            for ii, ss in enumerate(stn_ids):
                value = row[f"runoff_{ss}"]
                data += f" {value:.1f}"
            # <
            file.write(time + data + lb)

    # <<
    return None
