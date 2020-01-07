from functools import partial
from pathlib import Path
from typing import Dict

import click
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from multiprocessing import Process, Pool
import sklearn

from sklearn.svm import SVC
from xgboost import XGBClassifier
import pickle
from FeatureReader import HotEncodingReader as feature_reader


class ClassifierWithGridSearch (object):
    def __init__(self, dataset_file, result_dir, expected_num_of_features=580, features_to_remove=[]):
        self.dataset_file = dataset_file
        self.expected_num_of_features = expected_num_of_features
        self.features_to_remove = features_to_remove
        self.dataset_name = self.extract_dataset_name()
        print (f"Handling dataset : {self.dataset_name}")
        self.load_dataset()
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True, parents=True)
        self.create_clf_dict()

    def extract_dataset_name (self):
        return str(self.dataset_file.stem).split("_test")[0]

    def create_clf_dict (self):
        self.clf_dict = {
            'rf': RandomForestClassifier(),
            'SVM': SVC(),
            'logit': LogisticRegression(),
            'SGD': SGDClassifier(),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "xgbs": XGBClassifier(),
            "xgbs_no_encoding" : XGBClassifier(),
        }

    # def load_dataset(self, use_test_as_val: bool = False):
    #     directory = self.dataset_file.parent
    #     suffixes = ["train", "test"] if use_test_as_val else ["train_train", "train_val"]
    #     train_key = suffixes[0]
    #     val_key = suffixes[1]
    #
    #     keys = [f"{s}" for s in suffixes]
    #     df_dict = {key: pd.read_csv(directory / f"{self.dataset_name}_{key}.csv") for key in keys}
    #
    #     data = pd.concat([df_dict[train_key], df_dict[val_key]], axis=0)
    #     data.reset_index(inplace=True, drop=True)
    #     val_idx = np.concatenate(
    #         ((-1) * np.ones(df_dict[train_key].shape[0]), np.zeros(df_dict[val_key].shape[0])))
    #     ps = PredefinedSplit(val_idx)
    #     X = data.drop(columns=["Label", "microRNA_name"])
    #     expected_num_of_features = 580
    #     assert len(X.columns) == expected_num_of_features, f"""Read error. Wrong number of features.
    #            Read: {len(X.columns)}
    #            Expected: {expected_num_of_features}"""
    #     y = data.Label.ravel()
    #     train_index, val_index = next(ps.split())
    #     X_val = X.iloc[val_index]
    #     y_val = y[val_index]
    #     self.X = X
    #     self.y = y
    #     self.X_val = X_val
    #     self.y_val = y_val
    #     self.ps = ps

    def load_dataset(self):
        directory = self.dataset_file.parent
        X, y = read_feature_csv(directory / f"{self.dataset_name}_train.csv",
                                expected_num_of_features=self.expected_num_of_features,
                                features_to_remove=self.features_to_remove)

        assert len(X.columns) == self.expected_num_of_features, f"""Read error. Wrong number of features.
            Read: {len(X.columns)}
            Expected: {self.expected_num_of_features}"""

        self.X = X
        self.y = y


    def train_one_conf (self, clf_name, conf, scoring="accuracy"):
        output_file = self.result_dir / f"{self.dataset_name}_{clf_name}.csv"
        if output_file.is_file():
            print (f"output file: {output_file} exits. skip.")
            return

        clf = self.clf_dict[clf_name]
        print (clf)
        parameters = conf['parameters']

        grid_obj = GridSearchCV(clf, parameters, scoring=scoring, cv=4, n_jobs=-1, verbose=3)
        grid_obj.fit(self.X, self.y)

        print('\n Best estimator:')
        print(grid_obj.best_estimator_)
        print(grid_obj.best_score_ * 2 - 1)
        #save the best classifier
        best_clf = grid_obj.best_estimator_
        model_file = self.result_dir / f"{self.dataset_name}_{clf_name}.model"

        try:
            with model_file.open("wb") as pfile:
                pickle.dump(best_clf, pfile)
        except Exception:
            pass


        results = pd.DataFrame(grid_obj.cv_results_)
        results.to_csv(output_file, index=False)

    def fit_best_clf (self, clf_name, parameters: Dict):
        clf = self.clf_dict[clf_name]
        print (clf)
        fit_params = {}
        if clf_name=="xgbs":
            fit_params = {"eval_set": [(self.X_val, self.y_val)],
                          "early_stopping_rounds": 50}

        clf.fit(self.X, self.y, **fit_params)


        return clf

    def fit (self, yaml_path):
        with open(yaml_path, 'r') as stream:
            training_config = yaml.safe_load(stream)

        for clf_name, conf in training_config.items():
            if conf["run"]:
                self.train_one_conf(clf_name, conf, scoring="accuracy")


def worker (dataset_file, results_dir, yaml_file, expected_num_of_features=580, features_to_remove=[]):
    clf_grid_search = ClassifierWithGridSearch(dataset_file=dataset_file, result_dir=results_dir,
                                               expected_num_of_features=expected_num_of_features,
                                               features_to_remove=features_to_remove)
    clf_grid_search.fit(yaml_file)
    return



@click.group()
def cli():
    pass

@click.command()
@click.argument('yaml_file')
@click.argument('first_self', type=int)
@click.argument('last_self', type=int)
def self_fit(yaml_file, first_self, last_self):
    csv_dir = Path("Features/CSV")
    process_list = []
    p = Pool(24)
    for i in range(first_self, last_self):
        train_test_dir = csv_dir / f"train_test{i}"
        results_dir = Path("Results") / f"self{i}"

        partial_worker = partial(worker, results_dir=results_dir, yaml_file=yaml_file)
        p.map(partial_worker, train_test_dir.glob("*_test*"))


        # for dataset_file in train_test_dir.glob("*_test*"):
        #     worker(dataset_file, results_dir, yaml_file)

@click.command()
@click.argument('yaml_file')
@click.argument('first_self', type=int)
@click.argument('last_self', type=int)
def self_fit_no_encoding(yaml_file, first_self, last_self):
    csv_dir = Path("Features/CSV")
    features_to_remove = ['HotPairingMRNA_he_P1_L1',
                         'HotPairingMRNA_he_P1_L2',
                         'HotPairingMRNA_he_P1_L3',
                         'HotPairingMRNA_he_P1_L4',
                         'HotPairingMRNA_he_P1_L5',
                         'HotPairingMRNA_he_P2_L1',
                         'HotPairingMRNA_he_P2_L2',
                         'HotPairingMRNA_he_P2_L3',
                         'HotPairingMRNA_he_P2_L4',
                         'HotPairingMRNA_he_P2_L5',
                         'HotPairingMRNA_he_P3_L1',
                         'HotPairingMRNA_he_P3_L2',
                         'HotPairingMRNA_he_P3_L3',
                         'HotPairingMRNA_he_P3_L4',
                         'HotPairingMRNA_he_P3_L5',
                         'HotPairingMRNA_he_P4_L1',
                         'HotPairingMRNA_he_P4_L2',
                         'HotPairingMRNA_he_P4_L3',
                         'HotPairingMRNA_he_P4_L4',
                         'HotPairingMRNA_he_P4_L5',
                         'HotPairingMRNA_he_P5_L1',
                         'HotPairingMRNA_he_P5_L2',
                         'HotPairingMRNA_he_P5_L3',
                         'HotPairingMRNA_he_P5_L4',
                         'HotPairingMRNA_he_P5_L5',
                         'HotPairingMRNA_he_P6_L1',
                         'HotPairingMRNA_he_P6_L2',
                         'HotPairingMRNA_he_P6_L3',
                         'HotPairingMRNA_he_P6_L4',
                         'HotPairingMRNA_he_P6_L5',
                         'HotPairingMRNA_he_P7_L1',
                         'HotPairingMRNA_he_P7_L2',
                         'HotPairingMRNA_he_P7_L3',
                         'HotPairingMRNA_he_P7_L4',
                         'HotPairingMRNA_he_P7_L5',
                         'HotPairingMRNA_he_P8_L1',
                         'HotPairingMRNA_he_P8_L2',
                         'HotPairingMRNA_he_P8_L3',
                         'HotPairingMRNA_he_P8_L4',
                         'HotPairingMRNA_he_P8_L5',
                         'HotPairingMRNA_he_P9_L1',
                         'HotPairingMRNA_he_P9_L2',
                         'HotPairingMRNA_he_P9_L3',
                         'HotPairingMRNA_he_P9_L4',
                         'HotPairingMRNA_he_P9_L5',
                         'HotPairingMirna_he_P1_L1',
                         'HotPairingMirna_he_P1_L2',
                         'HotPairingMirna_he_P1_L3',
                         'HotPairingMirna_he_P1_L4',
                         'HotPairingMirna_he_P1_L5',
                         'HotPairingMirna_he_P2_L1',
                         'HotPairingMirna_he_P2_L2',
                         'HotPairingMirna_he_P2_L3',
                         'HotPairingMirna_he_P2_L4',
                         'HotPairingMirna_he_P2_L5',
                         'HotPairingMirna_he_P3_L1',
                         'HotPairingMirna_he_P3_L2',
                         'HotPairingMirna_he_P3_L3',
                         'HotPairingMirna_he_P3_L4',
                         'HotPairingMirna_he_P3_L5',
                         'HotPairingMirna_he_P4_L1',
                         'HotPairingMirna_he_P4_L2',
                         'HotPairingMirna_he_P4_L3',
                         'HotPairingMirna_he_P4_L4',
                         'HotPairingMirna_he_P4_L5',
                         'HotPairingMirna_he_P5_L1',
                         'HotPairingMirna_he_P5_L2',
                         'HotPairingMirna_he_P5_L3',
                         'HotPairingMirna_he_P5_L4',
                         'HotPairingMirna_he_P5_L5',
                         'HotPairingMirna_he_P6_L1',
                         'HotPairingMirna_he_P6_L2',
                         'HotPairingMirna_he_P6_L3',
                         'HotPairingMirna_he_P6_L4',
                         'HotPairingMirna_he_P6_L5',
                         'HotPairingMirna_he_P7_L1',
                         'HotPairingMirna_he_P7_L2',
                         'HotPairingMirna_he_P7_L3',
                         'HotPairingMirna_he_P7_L4',
                         'HotPairingMirna_he_P7_L5',
                         'HotPairingMirna_he_P8_L1',
                         'HotPairingMirna_he_P8_L2',
                         'HotPairingMirna_he_P8_L3',
                         'HotPairingMirna_he_P8_L4',
                         'HotPairingMirna_he_P8_L5',
                         'HotPairingMirna_he_P9_L1',
                         'HotPairingMirna_he_P9_L2',
                         'HotPairingMirna_he_P9_L3',
                         'HotPairingMirna_he_P9_L4',
                         'HotPairingMirna_he_P9_L5']
    expected_num_of_features = 580 - 90
    p = Pool(1)
    for i in range(first_self, last_self):
        train_test_dir = csv_dir / f"train_test{i}"
        results_dir = Path("Results") / f"self{i}"
        for t in  train_test_dir.glob("*_test*"):
            worker(dataset_file=t, results_dir=results_dir, yaml_file=yaml_file,
                                 expected_num_of_features=expected_num_of_features,
                                 features_to_remove=features_to_remove)


        # for dataset_file in train_test_dir.glob("*_test*"):
        #     worker(dataset_file, results_dir, yaml_file)



@click.command()
@click.argument('yaml_file')
@click.argument('first_self', type=int)
@click.argument('last_self', type=int)
def self_fit_random(yaml_file, first_self, last_self):
    csv_dir = Path("Features/CSV")
    process_list = []
    for i in range(first_self, last_self):
        train_test_dir = csv_dir / f"random_train_test{i}"
        results_dir = Path("Results") / f"random_self{i}"
        for dataset_file in train_test_dir.glob("*_test*"):
            worker(dataset_file, results_dir, yaml_file)



        #     p = Process(target=worker, args=(dataset_file, results_dir, yaml_file))
        # #     p.start()
        #
        #     process_list.append(p)
        # for p in process_list:
        #     p.join()

@click.command()
@click.argument('yaml_file')
@click.argument('first_self', type=int)
@click.argument('last_self', type=int)
def different_fit(yaml_file, first_self, last_self):
    csv_dir = Path("Features/CSV")
    for i in range(first_self, last_self):
        train_test_dir = csv_dir / f"train_test{i}"
        results_dir = Path("Results") / f"self{i}" / "xgbs_different"
        results_dir.mkdir(exist_ok=True)
        for dataset_file in train_test_dir.glob("*test*"):
            worker(dataset_file, results_dir, yaml_file, True)


cli.add_command(self_fit)
cli.add_command(different_fit)
cli.add_command(self_fit_random)
cli.add_command(self_fit_no_encoding)





if __name__ == '__main__':
    cli()
