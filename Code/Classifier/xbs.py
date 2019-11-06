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

Skip
to
content
Why
GitHub?
Enterprise
Explore
Marketplace
Pricing
Search

Sign in
Sign
up
15
398
30
ypeleg / HungaBunga
Code
Issues
5
Pull
requests
1
Projects
0
Security
Insights
Join
GitHub
today
GitHub is home
to
over
40
million
developers
working
together
to
host and review
code, manage
projects, and build
software
together.

HungaBunga / hunga_bunga / classification.py
Yam
Peleg
hunga
bunga
f1f13c6
14
days
ago
212
lines(169
sloc)  7.99
KB

import warnings

warnings.filterwarnings('ignore')

import numpy as np
from sklearn import datasets
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron, PassiveAggressiveClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, Matern, StationaryKernelMixin, WhiteKernel
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import is_classifier

from core import *
from params import *

linear_models_n_params = [
    (SGDClassifier,
     {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
      'alpha': [0.0001, 0.001, 0.1],
      'penalty': penalty_12none
      }),

    (LogisticRegression,
     {'penalty': penalty_12, 'max_iter': max_iter, 'tol': tol, 'warm_start': warm_start, 'C': C, 'solver': ['liblinear']
      }),

    (Perceptron,
     {'penalty': penalty_all, 'alpha': alpha, 'n_iter': n_iter, 'eta0': eta0, 'warm_start': warm_start
      }),

    (PassiveAggressiveClassifier,
     {'C': C, 'n_iter': n_iter, 'warm_start': warm_start,
      'loss': ['hinge', 'squared_hinge'],
      })
]

linear_models_n_params_small = linear_models_n_params

svm_models_n_params = [
    (SVC,
     {'C': C, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol,
      'max_iter': max_iter_inf2}),

    (NuSVC,
     {'nu': nu, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol
      }),

    (LinearSVC,
     {'C': C, 'penalty_12': penalty_12, 'tol': tol, 'max_iter': max_iter,
      'loss': ['hinge', 'squared_hinge'],
      })
]

svm_models_n_params_small = [
    (SVC,
     {'C': C, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol,
      'max_iter': max_iter_inf2}),

    (NuSVC,
     {'nu': nu, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol
      }),

    (LinearSVC,
     {'C': C, 'penalty': penalty_12, 'tol': tol, 'max_iter': max_iter,
      'loss': ['hinge', 'squared_hinge'],
      })
]

neighbor_models_n_params = [

    (KMeans,
     {'algorithm': ['auto', 'full', 'elkan'],
      'init': ['k-means++', 'random']}),

    (KNeighborsClassifier,
     {'n_neighbors': n_neighbors, 'algo': neighbor_algo, 'leaf_size': neighbor_leaf_size, 'metric': neighbor_metric,
      'weights': ['uniform', 'distance'],
      'p': [1, 2]
      }),

    (NearestCentroid,
     {'metric': neighbor_metric,
      'shrink_threshold': [1e-3, 1e-2, 0.1, 0.5, 0.9, 2]
      }),

    (RadiusNeighborsClassifier,
     {'radius': neighbor_radius, 'algo': neighbor_algo, 'leaf_size': neighbor_leaf_size, 'metric': neighbor_metric,
      'weights': ['uniform', 'distance'],
      'p': [1, 2],
      'outlier_label': [-1]
      })
]

gaussianprocess_models_n_params = [
    (GaussianProcessClassifier,
     {'warm_start': warm_start,
      'kernel': [RBF(), ConstantKernel(), DotProduct(), WhiteKernel()],
      'max_iter_predict': [500],
      'n_restarts_optimizer': [3],
      })
]

bayes_models_n_params = [
    (GaussianNB, {})
]

nn_models_n_params = [
    (MLPClassifier,
     {'hidden_layer_sizes': [(16,), (64,), (100,), (32, 32)],
      'activation': ['identity', 'logistic', 'tanh', 'relu'],
      'alpha': alpha, 'learning_rate': learning_rate, 'tol': tol, 'warm_start': warm_start,
      'batch_size': ['auto', 50],
      'max_iter': [1000],
      'early_stopping': [True, False],
      'epsilon': [1e-8, 1e-5]
      })
]

nn_models_n_params_small = [
    (MLPClassifier,
     {'hidden_layer_sizes': [(64,), (32, 64)],
      'batch_size': ['auto', 50],
      'activation': ['identity', 'tanh', 'relu'],
      'max_iter': [500],
      'early_stopping': [True],
      'learning_rate': learning_rate_small
      })
]

tree_models_n_params = [

    (RandomForestClassifier,
     {'criterion': ['gini', 'entropy'],
      'max_features': max_features, 'n_estimators': n_estimators, 'max_depth': max_depth,
      'min_samples_split': min_samples_split, 'min_impurity_split': min_impurity_split, 'warm_start': warm_start,
      'min_samples_leaf': min_samples_leaf,
      }),

    (DecisionTreeClassifier,
     {'criterion': ['gini', 'entropy'],
      'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
      'min_impurity_split': min_impurity_split, 'min_samples_leaf': min_samples_leaf
      }),

    (ExtraTreesClassifier,
     {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
      'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
      'min_impurity_split': min_impurity_split, 'warm_start': warm_start,
      'criterion': ['gini', 'entropy']})
]

tree_models_n_params_small = [

    (RandomForestClassifier,
     {'max_features_small': max_features_small, 'n_estimators_small': n_estimators_small,
      'min_samples_split': min_samples_split, 'max_depth_small': max_depth_small, 'min_samples_leaf': min_samples_leaf
      }),

    (DecisionTreeClassifier,
     {'max_features_small': max_features_small, 'max_depth_small': max_depth_small,
      'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf
      }),

    (ExtraTreesClassifier,
     {'n_estimators_small': n_estimators_small, 'max_features_small': max_features_small,
      'max_depth_small': max_depth_small,
      'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf})
]


def run_all_classifiers(x, y, small=True, normalize_x=True, n_jobs=cpu_count() - 1, brain=False, test_size=0.2,
                        n_splits=5, upsample=True, scoring=None, verbose=False, grid_search=True):
    all_params = (linear_models_n_params_small if small else linear_models_n_params) + (
        nn_models_n_params_small if small else nn_models_n_params) + (
                     [] if small else gaussianprocess_models_n_params) + neighbor_models_n_params + (
                     svm_models_n_params_small if small else svm_models_n_params) + (
                     tree_models_n_params_small if small else tree_models_n_params)
    return main_loop(all_params, StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=True,
                     n_jobs=n_jobs, verbose=verbose, brain=brain, test_size=test_size, n_splits=n_splits,
                     upsample=upsample, scoring=scoring, grid_search=grid_search)


class HungaBungaClassifier(ClassifierMixin):
    def __init__(self, brain=False, test_size=0.2, n_splits=5, random_state=None, upsample=True, scoring=None,
                 verbose=False, normalize_x=True, n_jobs=cpu_count() - 1, grid_search=True):
        self.model = None
        self.brain = brain
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.upsample = upsample
        self.scoring = None
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.normalize_x = normalize_x
        self.grid_search = grid_search
        super(HungaBungaClassifier, self).__init__()

    def fit(self, x, y):
        self.model = \
        run_all_classifiers(x, y, normalize_x=self.normalize_x, test_size=self.test_size, n_splits=self.n_splits,
                            upsample=self.upsample, scoring=self.scoring, verbose=self.verbose, brain=self.brain,
                            n_jobs=self.n_jobs, grid_search=self.grid_search)[0]
        return self

    def predict(self, x):
        return self.model.predict(x)


if __name__ == '__main__':
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    clf = HungaBungaClassifier()
    clf.fit(X, y)
    print(clf.predict(X).shape)

© 2019
GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact
GitHub
Pricing
API
Training
Blog
About

basic_training_config = {
    'rf': {
        "run": True,
        'parameters': {
            'n_estimators': [10, 50, 200, 500],
            'criterion': ['gini', 'entropy'],
            'max_depth': [2, 4, 10, 20],
            'min_samples_leaf': [2, 3],
            'max_features': ['auto', 'sqrt', 'log2'],
            "n_jobs": [-1],
        },
    },

    'SVM': {
        "run": True,
        'parameters': {
            'C': [0.5, 1, 2, 4, 16, 256],
            'kernel': ["linear", "poly", "rbf", "sigmoid"]
        }
    },
    'logit': {
        "run": True,
        'parameters': {
            'penalty': ['l1', 'l2'],
            'C': [round(0.2*x,2) for x in range(5,40)]
        }
    },
    'SGD': {
        "run": True,
        'parameters': {
            'penalty': ['l1', 'l2'],
            "n_jobs" : [-1],
            'loss': ['hinge', 'log'],
             'alpha': [0.0001, 0.001, 0.1],
        }
    },
    "KNeighborsClassifier": {
        "run" : True,
        'parameters': {
            'n_neighbors': [5, 10],
            'weights': ['uniform', 'distance'],
            "algorithm" : ['auto'],
            "n_jobs": [-1]
        }
    }
}