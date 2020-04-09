from pathlib import Path
from typing import Tuple

import pandas as pd


def mean_std(dir: Path, match: str, summary_shape: Tuple, summary_file_name="summary.csv"):
    results_dirs = [x for x in dir.iterdir() if x.is_dir() and x.match(match)]
    results_files = [x/summary_file_name for x in results_dirs]
    summary_df = []
    for f in results_files:
        try:
            summary_df.append(pd.read_csv(f, index_col=0))
        except FileNotFoundError:
            pass

    # summary_df = [pd.read_csv(f, index_col=0) for f in results_files]

    print (f"Num of summary files: {len(summary_df)}")
    print (f"shape = {summary_df[0].shape}")
    summary_df = [df for df in summary_df if df.shape==summary_shape]
    print (f"Num of valid summary files: {len(summary_df)}")
    # l = [x.iloc[1,1] for x in summary_df]

    summary_concat = pd.concat(summary_df)
    by_row_index = summary_concat.groupby(summary_concat.index)
    summary_means = by_row_index.mean()
    print (summary_means)
    summary_std = by_row_index.std(ddof=0)
    print (summary_std)
    result = summary_means.copy()
    for i in range (result.shape[0]):
        for j in range (result.shape[1]):
            m = round(summary_means.iloc[i,j], 3)
            s = round(summary_std.iloc[i,j], 3)
            result.iloc[i,j] = f"{m} ({s})"
    result.index = summary_df[0].index
    print(result)
    out_file = dir / f"{match}_{summary_file_name}_mean_std.csv"
    result.to_csv(out_file)
    print()
    print (f"Num of valid summary files: {len(summary_df)}")
    print()
    print (result.to_latex())

    return result, summary_means, summary_df
    train_test_dir_parent = Path("Features/CSV/")

    res_table = pd.DataFrame()

    for results_dir in results_dirs:
        print(results_dir)
