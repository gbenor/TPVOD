from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
import pandas as pd
import numpy as np
from pathlib import Path
from collections import ChainMap
from TPVOD_Utils import utils
from multiprocessing import Process

basic_training_config = {
    'rf': {
        "run": True,
        'clf': RandomForestClassifier(),
        'parameters': {
            'n_estimators': [10, 50, 200, 500],
            'criterion': ['gini', 'entropy'],
            'max_depth': [2, 4, 10, 20],
            'min_samples_leaf': [2, 3],
            'max_features': ['auto', 'sqrt', 'log2'],
            "n_jobs": [-1],
        },
    },
    "XGBoost": {
        "run" : True,
        'clf': XGBClassifier(),
        'parameters': {
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
        }
    },
    'SVM': {
        "run": True,
        'clf': SVC(),
        'parameters': {
            'C': [0.5, 1, 2, 4, 16, 256],
            'kernel': ["linear", "poly", "rbf", "sigmoid"]
        }
    },
    'logit': {
        "run": True,
        'clf': LogisticRegression(),
        'parameters': {
            'penalty': ['l1', 'l2'],
            'C': list(np.arange(0.5, 8.0, 0.1))
        }
    },
    'SGD': {
        "run": True,
        'clf': SGDClassifier(),
        'parameters': {
            'penalty': ['l1', 'l2'],
            "n_jobs" : [-1],
        }
    }
}



def grid_search (X, y, output_file, training_config=basic_training_config ,scoring="accuracy"):
    for name, conf in training_config.items():
        # put here in order to prevent us from rerun
        # TBD: delete this
        o = utils.filename_suffix_append(output_file, f"_{name}")
        o = Path(o)

        if o.is_file():
            continue
        # print(f"{o}  {o.is_file()}")
        # continue
        # TBD : delete the uppers code rows



        if not conf["run"]:
            continue
        print('=' * 20)
        print (f"=========== {name} =============")

        clf = conf['clf']
        parameters = conf['parameters']
        grid_obj = GridSearchCV(clf, parameters, scoring=scoring, cv=4, n_jobs=-1, verbose=3)
        grid_obj.fit(X, y)

        print('\n Best estimator:')
        print(grid_obj.best_estimator_)
        print(grid_obj.best_score_ * 2 - 1)
        results = pd.DataFrame(grid_obj.cv_results_)
        results.to_csv(utils.filename_suffix_append(output_file, f"_{name}"), index=False)

        # xgb = XGBClassifier()
        # skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 42)
        #
        # random_search = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_params, n_iter=10,
        #                                    scoring='accuracy', n_jobs=-1, cv=skf.split(X,y), verbose=2,
        #                                    random_state=1001)


def train_self():
    train_test_dir = Path("Features/CSV/train_test")
    results_dir = Path("Results") / "self"

    process_list = []

    for f in train_test_dir.glob("*self_train_all*"):
        if f.stat().st_size < 99999:
            continue
        dataset = str(f.stem).split("_self")[0]

        test                    = f"{dataset}_test.csv"
        self_train_fixed        = f"{dataset}_self_train_fixed.csv"
        self_train_all          = f"{dataset}_self_train_all.csv"
        different_train_fixed   = f"{dataset}_different_train_fixed.csv"
        different_train_all     = f"{dataset}_different_train_all.csv"

        pairs = [
            ("self_fixed", self_train_fixed, test),
            ("self_all", self_train_all, test)
        ]



        for title, train, test in pairs:

            df = pd.read_csv(train_test_dir/train)
            X =  df.drop (["Label"], axis=1)
            y = df.Label.ravel()

            output_file = results_dir / f"{dataset}_{title}_results.csv"
            p = Process(target=grid_search, args=(X, y, output_file))
            p.start()
            process_list.append(p)
            # grid_search(X, y, output_file)
    for p in process_list:
        p.join()



def train_different():
    csv_dir = Path("Features/CSV")
    for i in range(1, 21):
        train_test_dir = csv_dir / f"train_test{i}"
        results_dir = Path("Results") / f"different{i}"
        results_dir.mkdir()

        xgb_training_config = basic_training_config
        clfs = list(xgb_training_config.keys())
        for clf in clfs:
            if clf != "XGBoost":
                xgb_training_config[clf]["run"] = False




        for f in train_test_dir.glob("*different_train_fixed*"):
            if f.stat().st_size < 99999:
                continue
            dataset = str(f.stem).split("_different")[0]

            test                    = f"{dataset}_test.csv"
            self_train_fixed        = f"{dataset}_self_train_fixed.csv"
            self_train_all          = f"{dataset}_self_train_all.csv"
            different_train_fixed   = f"{dataset}_different_train_fixed.csv"
            different_train_all     = f"{dataset}_different_train_all.csv"

            pairs = [
                ("different_fixed", different_train_fixed, test),
                ("different_all", different_train_all, test)
            ]


            for title, train, test in pairs:

                df = pd.read_csv(train_test_dir/train)
                X =  df.drop (["Label"], axis=1)
                y = df.Label.ravel()

                output_file = results_dir / f"{dataset}_{title}_results.csv"

                grid_search(X, y, output_file, xgb_training_config)






def main():
    # train_self()
    train_different()


if __name__ == "__main__":
    main()

    # # bf = pd.read_csv(TestAgainstMyself_dir / "best_features.csv", index_col=0)
    # # for f_train in TestAgainstMyself_dir.glob('*test*'):
    #
    # # for f_train in TestAgainstMyself_dir.glob('*train_fixed*'):
    # for f_train in TestAgainstMyself_dir.glob('*'):
    #
    #     df = pd.read_csv(f_train, index_col=0)
    #     if df.shape[0] < 1000:
    #         continue
    #     print(f_train)
    #     y = df.Label.ravel()
    #     # X = df[bf.columns]
    #     for num_of_feature in [100, "all"]:
    #         X = SelectKBest(mutual_info_classif, k=num_of_feature).fit_transform(df.drop(["Label"], axis=1), y)
    #         # X = df
    #         # X.drop (["Label"], axis=1, inplace=True)
    #         output_file = results_dir / f"TestAgainstMyself_k={num_of_feature}_{f_train.stem}.csv"
    #         random_search(X, y, output_file)
    #
    #

