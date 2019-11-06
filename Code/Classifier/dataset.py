from pathlib import Path
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import PredefinedSplit


def drop_unnamed (df : DataFrame):
    return  df.loc[:, ~df.columns.str.contains('^Unnamed')]

class Dataset(object):
    def __init__(self, data_dict: dict):
        self.X, self.y, \
        self.X_val, self.y_val, \
        self.X_test, self.y_test, \
        self.ps = self.extract_data(data_dict)

    @classmethod
    def create_class_from_size_file(cls, size_train_train_file: Path):
        parent_dir = size_train_train_file.parent
        dataset = size_train_train_file.stem.split("_train_train")[0]
        data_dict = {"train_train" : pd.read_csv(size_train_train_file),
                   "train_val"   : pd.read_csv(parent_dir / f"{dataset}_train_val.csv"),
                   "test"        : pd.read_csv(parent_dir / f"{dataset}_test.csv")
                   }
        return cls(data_dict)


    @staticmethod
    def extract_Xy (data: DataFrame) -> (DataFrame, np.ndarray):
        X = data.drop(columns=["Label", "microRNA_name"])
        y = data["Label"].ravel()
        return X, y

    def extract_data(self, df_dict: dict):
        for key in df_dict.keys():
            df_dict[key] = drop_unnamed(df_dict[key])

        data = pd.concat([df_dict["train_train"], df_dict["train_val"]], axis=0)
        data.reset_index(inplace=True, drop=True)
        val_idx = np.concatenate(
            ((-1) * np.ones(df_dict["train_train"].shape[0]), np.zeros(df_dict["train_val"].shape[0])))
        ps = PredefinedSplit(val_idx)
        X, y = self.extract_Xy(data)

        train_index, val_index = next(ps.split())
        X_val = X.iloc[val_index]
        y_val = y[val_index]

        X_test, y_test = self.extract_Xy(df_dict["test"])
        assert set(X.columns) == set(X_test.columns), f"""X and X_test must have the same columns.
         {np.setdiff1d(set(X.columns), set(X_test.columns))}"""

        return X, y, X_val, y_val, X_test, y_test, ps
