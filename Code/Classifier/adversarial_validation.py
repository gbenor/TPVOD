import multiprocessing
import pprint
from itertools import combinations
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import xgboost as xgb
from multiprocessing import Process
from pandas import DataFrame
from classifier_utils import create_dataset_for_adversarial_validation, read_feature_csv, custom_gridsearch


def same_dataset_files(path: Path) -> List[List[Tuple]]:
    dataset_prefix = [f.stem.split("_train_train")[0] for f in path.glob("*dataset*_train_train.csv")]
    dataset = [[path / f"{d}_{suffix}.csv" for suffix in ["test", "train_val", "train_train"]] for d in dataset_prefix]
    all_combinations = [list(combinations(d, 2)) for d in dataset]
    return all_combinations

def different_dataset_files(path: Path) -> List[List[Tuple]]:
    # dataset_prefix = [f.stem.split("_train_train")[0] for f in path.glob("*dataset*_train_train.csv")]
    # dataset = [[path / f"{d}_{suffix}.csv" for suffix in ["test", "train_val", "train_train"]] for d in dataset_prefix]
    all_combinations = [list(combinations( path.glob("*pos*"), 2)) ]
    return all_combinations

def adv_validation_worker (pair: Tuple[Path, Path], return_dict: Dict[Tuple, float]):
    print(pair)

    d0 = pd.read_csv(pair[0])
    col_list = list(d0.columns)
    feature_index = col_list.index("Seed_match_compact_A")
    d0.drop(columns=col_list[:feature_index], inplace=True)

    d1 = pd.read_csv(pair[1])
    col_list = list(d1.columns)
    feature_index = col_list.index("Seed_match_compact_A")
    d1.drop(columns=col_list[:feature_index], inplace=True)

    # X, y = create_dataset_for_adversarial_validation(d0, d1, col_to_drop=['Label', 'microRNA_name'])
    X, y = create_dataset_for_adversarial_validation(d0, d1, col_to_drop=[])
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=44)
    xgb_params = {
        'learning_rate': 0.05, 'max_depth': 4, 'subsample': 0.9,
        'colsample_bytree': 0.9, 'objective': 'binary:logistic',
        'silent': 1, 'n_estimators': 100, 'gamma': 1,
        'min_child_weight': 4
    }
    clf = xgb.XGBClassifier(**xgb_params, seed=10)
    for train_index, test_index in skf.split(X, y):
        x0, x1 = X.iloc[train_index], X.iloc[test_index]
        y0, y1 = y[train_index], y[test_index]
        clf.fit(x0, y0, eval_set=[(x1, y1)],
                eval_metric='logloss', verbose=False, early_stopping_rounds=10)

        prval = clf.predict_proba(x1)[:, 1]
        return_dict[pair] = roc_auc_score(y1, prval)
        # print(f"AUC: {roc_auc_score(y1, prval)}")
        # results = pd.DataFrame()
        # results['columns'] = X.columns
        # results['importances'] = clf.feature_importances_
        # results.sort_values(by='importances', ascending=False, inplace=True)
        # print(results.head(5))
        # print()

def test_dataset_coherency_worker(data: DataFrame, dataset_name, conf_yaml: Path, result_dir: Path):
    with conf_yaml.open("r") as stream:
        conf =  yaml.safe_load(stream)
    d0, d1 = train_test_split(data, test_size=0.5)
    X, y = create_dataset_for_adversarial_validation(d0, d1, col_to_drop=[])
    clf_name = "xgbs"
    custom_gridsearch(clf_name, dataset_name, conf[clf_name], result_dir, "roc_auc", X, y, 3)


def test_dataset_coherency():
    dataset_dir = Path("Features/CSV")
    conf_yaml = Path("../Code/Classifier/yaml/xgbs_params.yml")
    result_dir = Path("Results/adversarial_validation/dataset")
    for pos_file in dataset_dir.glob("pos*.csv"):
        dataset_name = pos_file.stem.replace("pos_", "")
        neg_file = dataset_dir / (pos_file.stem.replace("pos", "neg") + ".csv")
        pos = read_feature_csv(pos_file)
        neg = read_feature_csv(neg_file)
        data = pd.concat([pos, neg])
        test_dataset_coherency_worker(data, dataset_name, conf_yaml, result_dir)


def test_dataset_coherency_result():
    results_dir = Path("Results/adversarial_validation/dataset")
    summary = pd.DataFrame()
    for i, res_file in enumerate(results_dir.glob("*_xgbs.csv")):
        name = res_file.stem.split("_xgbs")[0]
        print (f"file: {res_file}       name:{name}")

        df = pd.read_csv(res_file)
        best = df[df["rank_test_score"]==1].head(1)
        m = round(best["mean_test_score"].item(), 3)
        s = round(best["std_test_score"].item(), 3)
        summary.loc[name, "ROC-AUC"] = f"{m} ({s})"
        print (summary)
    summary.sort_index(inplace=True)
    print(summary)
    summary.to_csv(results_dir/ "summary.csv")
    print(60*"*")
    print(summary.to_latex())



def main():
    # test_dataset_coherency()
    test_dataset_coherency_result()

    # # path = Path("Features/CSV/train_test3")
    # # all_combinations = same_dataset_files(path)
    # path = Path("Features/CSV")
    # all_combinations = different_dataset_files(path)
    # process_list = []
    # manager = multiprocessing.Manager()
    # return_dict = manager.dict()
    # for dataset_pairs in all_combinations:
    #     for pair in dataset_pairs:
    #         p = Process(target=adv_validation_worker, args=(pair, return_dict))
    #         p.start()
    #         process_list.append(p)
    # for p in process_list:
    #     p.join()
    #
    # result_df = pd.DataFrame()
    # for key, value in return_dict.items():
    #     result_df.loc[key[0].stem, key[1].stem] = value
    #
    #
    #
    # print(result_df.to_latex())
    # result_df.to_csv("result_df.csv")
    #



if __name__ == '__main__':
    main()


