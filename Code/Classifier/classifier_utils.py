from pathlib import Path
from typing import List, Tuple, Union
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import sklearn
from xgboost import XGBClassifier

CLF_DICT = {
    'rf': RandomForestClassifier(),
    'SVM': SVC(),
    'logit': LogisticRegression(),
    'SGD': SGDClassifier(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "xgbs" : XGBClassifier(),
}


def create_dataset_for_adversarial_validation (dataset1: DataFrame, dataset2: DataFrame, col_to_drop: List[str]=[]) \
        -> Tuple[DataFrame, ndarray]:
    X: DataFrame = pd.concat([dataset1, dataset2], ignore_index=True)
    X.drop(columns=col_to_drop, inplace=True)
    y: ndarray = np.concatenate([np.ones(dataset1.shape[0]), np.zeros(dataset2.shape[0])])
    assert X.shape[0] == y.shape[0], f"X and y must be at the same length. X={X.shape}  y={y.shape}"
    return X, y

def custom_gridsearch (clf_name: str, dataset_name: str, conf:dict, result_dir: Path, scoring: str,
                       X: DataFrame, y:ndarray, ps: Union[int,sklearn.model_selection._split.PredefinedSplit]):
    output_file = result_dir / f"{dataset_name}_{clf_name}.csv"

    if output_file.is_file():
        print (f"output file: {output_file} exits. skip.")
        return

    clf = CLF_DICT[clf_name]
    print (clf)
    parameters = conf['parameters']
    try:
        fit_params = conf['fit_params']
    except KeyError:
        fit_params = {}


    grid_obj = GridSearchCV(clf, parameters, scoring=scoring, cv=ps, n_jobs=-1, verbose=3)
    grid_obj.fit(X, y, **fit_params)

    print('\n Best estimator:')
    print(grid_obj.best_estimator_)
    print(grid_obj.best_score_ * 2 - 1)
    results = pd.DataFrame(grid_obj.cv_results_)
    results.to_csv(output_file, index=False)

def read_feature_csv(f: Path) -> DataFrame:
    data: DataFrame = pd.read_csv(f)
    col_list = list(data.columns)
    feature_index = col_list.index("Seed_match_compact_A")
    data.drop(columns=col_list[:feature_index], inplace=True)
    expected_num_of_features = 580
    assert len(data.columns)==expected_num_of_features, f"""Read error. Wrong number of features.
    Read: {len(data.columns)}
    Expected: {expected_num_of_features}"""
    return data








