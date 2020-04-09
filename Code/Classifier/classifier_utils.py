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
from sklearn.metrics import confusion_matrix, accuracy_score

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
#
# def read_feature_csv(f: Path, data=None, expected_num_of_features = 580, features_to_remove=[]) -> DataFrame:
#     if data is None:
#         data: DataFrame = pd.read_csv(f)
#     y = data.Label.ravel()
#     col_list = list(data.columns)
#     feature_index = col_list.index("Seed_match_compact_A")
#     data.drop(columns=col_list[:feature_index], inplace=True)
#     if len(features_to_remove)>0:
#         data.drop(columns=features_to_remove, inplace=True)
#
#     assert len(data.columns)==expected_num_of_features, f"""Read error. Wrong number of features.
#     Read: {len(data.columns)}
#     Expected: {expected_num_of_features}"""
#     return data, y

#


def read_feature_csv(f: Path, expected_num_of_features = 580) -> (DataFrame, ndarray):
    data: DataFrame = pd.read_csv(f)
    y = data.Label.ravel()
    col_list = list(data.columns)
    feature_index = col_list.index("Seed_match_compact_A")
    data.drop(columns=col_list[:feature_index], inplace=True)

    assert len(data.columns)==expected_num_of_features, f"""Read error. Wrong number of features.
    Read: {len(data.columns)}
    Expected: {expected_num_of_features}"""
    return data, y




def clf_performance_report(y_true, y_pred):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    report = {
        # Sensitivity, hit rate, recall, or true positive rate
        "TPR" : TP / (TP + FN),
        # Specificity or true negative rate
        "TNR" : TN / (TN + FP),
        # Precision or positive predictive value
        "PPV" : TP / (TP + FP),
        # Negative predictive value
        "NPV" : TN / (TN + FN),
        # Fall out or false positive rate
        "FPR" : FP / (FP + TN),
        # False negative rate
        "FNR" : FN / (TP + FN),
        # False discovery rate
        "FDR" : FP / (TP + FP),

        # Overall accuracy
        "ACC" : (TP + TN) / (TP + FP + FN + TN),
    }
    assert report["ACC"] == accuracy_score(y_true, y_pred)
    return report








