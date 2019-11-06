from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import PredefinedSplit
import ast

import pandas as pd
import numpy as np
from pathlib import Path
from collections import ChainMap
from TPVOD_Utils import utils
from multiprocessing import Process

from dataset import Dataset

XGBS_PARAMS = {
            "objective": ["binary:hinge"],
            "booster" : ["gbtree"],
            "eta" : [0.1, 0.02, 0.3, 0.7],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'max_depth': range(2, 10, 2),
            'min_child_weight': [1, 5, 10],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            "lambda" : [1,2],
            "n_jobs": [-1],
            "verbose": [0],
    "verbose_eval ": [False]

}
#
#
# XGBS_PARAMS = {
#             "objective": ["binary:hinge"],
#             "booster" : ["gbtree"],
#             "eta" : [0.1],
#             'gamma': [0.5],
#             'max_depth': range(2, 4, 2),
#             'min_child_weight': [1],
#             'subsample': [0.6],
#             'colsample_bytree': [0.6],
#             "lambda" : [1],
#             "n_jobs": [-1],
#
#         }
#



def train_self(scoring='accuracy'):
    csv_dir = Path("Features/CSV")
    for i in range(10):
        train_test_dir = csv_dir / f"train_test{i}"
        results_dir = Path("Results") / f"self{i}"
        results_dir.mkdir(exist_ok=True)
        for dataset_file in train_test_dir.glob("*test*"):
            dataset = str(dataset_file.stem).split("_test")[0]
            suffixes = ["train", "train_train", "train_val"]
            keys = [f"{s}" for s in suffixes]
            df_dict = {key: pd.read_csv(train_test_dir/f"{dataset}_{key}.csv") for key in keys}

            #xgboost with eval
            ###################################
            data = pd.concat([df_dict["train_train"], df_dict["train_val"]], axis=0)
            data.reset_index(inplace=True, drop=True)
            val_idx = np.concatenate(((-1)*np.ones(df_dict["train_train"].shape[0]), np.zeros(df_dict["train_val"].shape[0])))
            ps = PredefinedSplit(val_idx)
            X = data.drop(columns=["Label", "microRNA_name"])
            y = data.Label.ravel()
            train_index, val_index = next(ps.split())
            X_val = X.iloc[val_index]
            y_val = y[val_index]

            output_file = results_dir / f"{dataset}_xgbs_val_results.csv"
            print(output_file)
            if not output_file.exists():
                clf = XGBClassifier(silent=True)
                grid_obj = GridSearchCV(clf, XGBS_PARAMS, scoring=scoring, cv=ps, verbose=3)
                fit_params = {"eval_set": [(X_val, y_val)],
                              "early_stopping_rounds": 50}
                grid_obj.fit(X, y, **fit_params)

                print('\n Best estimator:')
                print(grid_obj.best_estimator_)
                print(grid_obj.best_score_ * 2 - 1)
                results = pd.DataFrame(grid_obj.cv_results_)
                results.to_csv(output_file, index=False)

            # # xgboost without eval
            # ###################################
            # data = df_dict["train"]
            # data.reset_index(inplace=True, drop=True)
            # X = data.drop(columns=["Label", "microRNA_name"])
            # y = data.Label.ravel()
            #
            # output_file = results_dir / f"{dataset}_xgbs_all_results.csv"
            # clf = XGBClassifier()
            # grid_obj = GridSearchCV(clf, XGBS_PARAMS, scoring=scoring, cv=3, verbose=3)
            # grid_obj.fit(X, y)
            #
            # print('\n Best estimator:')
            # print(grid_obj.best_estimator_)
            # print(grid_obj.best_score_ * 2 - 1)
            # results = pd.DataFrame(grid_obj.cv_results_)
            # results.to_csv(output_file, index=False)
            #



def results_summary():
    results_dirs = [x for x in Path("Results").iterdir() if x.is_dir() and x.match("*self*")]
    train_test_dir_parent = Path("Features/CSV/")

    res_table = pd.DataFrame()

    for results_dir in results_dirs:
        print (results_dir)
        id = str(results_dir).split("self")[1]
        train_test_dir = train_test_dir_parent / f"train_test{id}"

        for i, f_res in enumerate(results_dir.glob("*results*.csv")):
            print (f_res)
            s = f_res.stem
            train_dataset = s.split("_xgbs")[0]
            train_dataset_type = "val" if f_res.match("*val*") else "all"

            df = pd.read_csv(f_res)
            best = df[df["rank_test_score"]==1]
            params = (best.head(1)["params"]).item()
            params =  ast.literal_eval(params)

            suffixes = ["train", "train_train", "train_val", "test"]
            keys = [f"{s}" for s in suffixes]
            df_dict = {key: pd.read_csv(train_test_dir / f"{train_dataset}_{key}.csv") for key in keys}

            test = df_dict["test"]
            X_test = test.drop(columns=["Label", "microRNA_name"])
            y_test = test.Label.ravel()

            if train_dataset_type=="val":
                # xgboost with eval
                ###################################
                data = pd.concat([df_dict["train_train"], df_dict["train_val"]], axis=0)
                data.reset_index(inplace=True, drop=True)
                val_idx = np.concatenate(
                    ((-1) * np.ones(df_dict["train_train"].shape[0]), np.zeros(df_dict["train_val"].shape[0])))
                ps = PredefinedSplit(val_idx)
                X = data.drop(columns=["Label", "microRNA_name"])
                y = data.Label.ravel()
                train_index, val_index = next(ps.split())
                X_val = X.iloc[val_index]
                y_val = y[val_index]

                model = XGBClassifier(**params)
                print(model)
                fit_params = {"eval_set": [(X_val, y_val)],
                              "early_stopping_rounds": 50}
                model.fit(X, y, **fit_params)
            else:
                # xgboost without eval
                ###################################
                data = df_dict["train"]
                data.reset_index(inplace=True, drop=True)
                X = data.drop(columns=["Label", "microRNA_name"])
                y = data.Label.ravel()

                model = XGBClassifier(**params)
                print(model)
                model.fit(X, y)



            score =  model.score(X_test, y_test)

            res_table.loc[f"{train_dataset_type}", f"{train_dataset}"] = round(score, 3)
            print(res_table)
        #
        #     # if i > 7:
        #     #     break
        res_table.sort_index(axis=0,  inplace=True)
        res_table.sort_index(axis=1,  inplace=True)

        print(res_table)
        print(res_table.to_latex())

        res_table.to_csv(results_dir/"summary.csv")




def train_size(scoring='accuracy'):
    csv_dir = Path("Features/CSV")
    train_test_dir = csv_dir / f"train_test_size"
    results_dir = Path("Results") / "train_test_size"
    results_dir.mkdir(exist_ok=True)
    for train_file in train_test_dir.glob("*train_train_*"):
        print (f"train file: {train_file}")
        ds = Dataset.create_class_from_size_file(train_file)
        size_str = str(train_file.stem).split("train_train_")[1]
        output_file = results_dir / f"{size_str}_results.csv"

        # xgboost with eval
        ###################################
        print(output_file)
        if not output_file.exists():
            clf = XGBClassifier(silent=True)
            grid_obj = GridSearchCV(clf, XGBS_PARAMS, scoring=scoring, cv=ds.ps, verbose=3)
            fit_params = {"eval_set": [(ds.X_val, ds.y_val)],
                          "early_stopping_rounds": 50}
            grid_obj.fit(ds.X, ds.y, **fit_params)

            print('\n Best estimator:')
            print(grid_obj.best_estimator_)
            print(grid_obj.best_score_ * 2 - 1)
            results = pd.DataFrame(grid_obj.cv_results_)
            results.to_csv(output_file, index=False)







def main():
    # train_self()
    train_size()
    # results_summary()



if __name__ == "__main__":
    main()
