import pickle
from functools import partial
from itertools import combinations
from multiprocessing import Process
from typing import List, Tuple
import matplotlib.pyplot as plt

import pandas as pd
from pathlib import Path
from collections import ChainMap

from seaborn import heatmap
from sklearn.metrics import accuracy_score, confusion_matrix

from ClassifierWithGridSearch import ClassifierWithGridSearch
import FeatureReader
from FeatureReader import get_reader

from TPVOD_Utils import utils
import json
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import PredefinedSplit
import ast
from multiprocessing import Pool

import pandas as pd
import numpy as np

from dataset import Dataset
from pandas import DataFrame


DATASET_LIST = ['human_dataset1', 'human_dataset2', 'human_dataset3', 'mouse_dataset1',  'mouse_dataset2',
                'celegans_dataset1', 'celegans_dataset2', 'cattle_dataset1']
DATASET_LIST.sort()



class NoModelFound(Exception):
    pass


def measurement (y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]

    d =  {
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
        # "ROC_AUC" : roc_auc_score(y_true, y_score
    }
    return {k: round(v,3) for k, v in d.items()}






def run_test (dataset, train_cfg, best_params, method, train_test_dir = Path("Features/CSV/train_test0")):
    f_test = train_test_dir / f"{dataset}_test.csv"
    test = pd.read_csv(f_test)
    X_test = test.drop(["Label"], axis=1)
    y_test = test.Label.ravel()


    f_train = train_test_dir / f"{dataset}_self_train_{train_cfg}.csv"


    train = pd.read_csv(f_train)
    X_train = train.drop(["Label"], axis=1)
    y_train = train.Label.ravel()

    model = None
    print (method)
    if method=="rf" :
        model = RandomForestClassifier(**best_params)
    elif method=="SVM" :
        model = SVC (**best_params)
    elif method=="logit" :
        model = LogisticRegression(**best_params)
    elif method == "SGD":
        model = SGDClassifier(**best_params)
    elif method == "XGBoost":
        model = XGBClassifier(**best_params)
    else:
        raise Exception("No classifer")
    print (model)
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

def get_presaved_clf(results_dir: Path, dataset: str, method: str):
    clf_file = results_dir / f"{dataset}_{method}.model"
    if not clf_file.is_file():
        raise NoModelFound(f"No model found: {clf_file}")
    with clf_file.open("rb") as f:
        clf = pickle.load(f)
    return clf

def self_results_summary(id: int, prefix=""):
    ms_table = None
    results_dir = Path("Results") / f"{prefix}self{id}"
    train_test_dir = Path(f"Features/CSV/{prefix}train_test{id}")
    methods = ['xgbs_no_encoding', 'rf', 'KNeighborsClassifier', 'SGD', 'SVM', 'logit']
    res_table: DataFrame = pd.DataFrame()
    feature_reader = get_reader()

    for f_test in train_test_dir.glob("*test*"):
        f_stem = f_test.stem
        dataset = f_stem[:f_stem.find("dataset") + 8]
        for method in methods:
            print(f"test: {f_test}, method: {method}")
            try:
                clf = get_presaved_clf(results_dir, dataset,method)
                X_test, y_test = feature_reader.file_reader(f_test)

                test_score = accuracy_score(y_test, clf.predict(X_test))

                res_table.loc[dataset, method] = round(test_score, 3)

                print(res_table)
                res_table.to_csv(results_dir / "summary.csv")
                if method in ["xgbs_no_encoding", "xgbs"]:
                    feature_importance = xgbs_feature_importance(clf, X_test)
                    feature_importance.to_csv(results_dir / f"feature_importance_{dataset}.csv")
                    print("save feature importance file")
                    ms =  measurement(y_test, clf.predict(X_test))
                    if ms_table is None:
                        ms_table = pd.DataFrame(columns=list(ms.keys()))
                    ms_table.loc[dataset] = ms
                    ms_table.to_csv(results_dir / "xgbs_measurements.csv")
            except NoModelFound:
                pass

    res_table.sort_index(inplace=True)
    print(res_table)
    print(res_table.to_latex())

    res_table.to_csv(results_dir/"summary.csv")

def self_results_summary_model(id: int):
    results_dir = Path("Results") / f"self{id}"
    train_test_dir = Path(f"Features/CSV/train_test{id}")

    res_table: DataFrame = pd.DataFrame()

    for f_model in results_dir.glob("*.model"):
        print (f"model: {f_model}")

        with f_model.open("rb") as f:
            best_clf = pickle.load(f)

        dataset =  "human_dataset1"
        method = "xgbs"
        f_test = train_test_dir / "human_dataset1_test.csv"

        X_test, y_test = read_feature_csv(f_test)
   #     y_test = pd.read_csv(f_test).Label.ravel()
        test_score = accuracy_score(y_test, best_clf.predict(X_test))

        res_table.loc[dataset, method] = round(test_score, 3)

        print(res_table)
        res_table.to_csv(results_dir / "summary.csv")
        feature_importance = xgbs_feature_importance(best_clf, X_test)
        feature_importance.to_csv(results_dir/ "feature_importance.csv")
        print ("save feature importance file")
        return measurement(y_test, best_clf.predict(X_test))

    res_table.sort_index(inplace=True)
    print(res_table)
    print(res_table.to_latex())

    res_table.to_csv(results_dir/"summary.csv")



def different_results_summary(id: int):
    def get_test_dataset(name, test_file_dir=Path("Features/CSV"), random_state=22):
        pos_file = test_file_dir / f"{name}_duplex_positive_feature.csv"
        neg_file = test_file_dir / f"{name}_duplex_negative_feature.csv"

        pos = pd.read_csv(pos_file)
        pos.insert(0, "Label", 1)
        neg = pd.read_csv(neg_file)
        neg.insert(0, "Label", 0)
        # Both dataset must have the same columns
        col = [c for c in pos.columns if c in neg.columns]
        pos = pos[col]
        neg = neg[col]

        test = pos.append(neg, ignore_index=True)
        return test.reindex(np.random.RandomState(seed=random_state).permutation(test.index))


    results_dir = Path("Results") / f"self{id}"
    train_test_dir = Path(f"Features/CSV/train_test{id}")

    clf_datasets = [f.stem.split("_xgbs")[0] for f in results_dir.glob("*_xgbs*model")]
    res_table = pd.DataFrame()
    methods = ["xgbs_no_encoding"]
    res_table: DataFrame = pd.DataFrame()
    feature_reader = get_reader()

    for clf_dataset in clf_datasets:
        for test_dataset in DATASET_LIST:
            for method in methods:
                print(f"clf: {clf_dataset}    test: {test_dataset}, method: {method}")
                try:
                    clf = get_presaved_clf(results_dir, clf_dataset, method)
                    if test_dataset != clf_dataset:
                        # X_test, y_test = read_feature_csv(f=None, data=get_test_dataset(test_dataset))
                        X_test, y_test = feature_reader.df_reader(get_test_dataset(test_dataset))

                    else:
                        # X_test, y_test = read_feature_csv(f=train_test_dir/f"{test_dataset}_test.csv")
                        X_test, y_test = feature_reader.file_reader(train_test_dir/f"{test_dataset}_test.csv")

                    test_score = accuracy_score(y_test, clf.predict(X_test))

                    res_table.loc[clf_dataset, test_dataset] = round(test_score, 3)
                    print(res_table)
                    res_table.to_csv(results_dir / "diff_summary.csv")
                except NoModelFound:
                    pass

    res_table.sort_index(axis=0, inplace=True)
    res_table.sort_index(axis=1, inplace=True)

    print(res_table)
    print(res_table.to_latex())
    res_table.to_csv(results_dir / "diff_summary.csv")

    # for results_dir in results_dirs:
    #     print (results_dir)
    #     id = str(results_dir).split("different")[1]
    #     train_test_dir = train_test_dir_parent / f"train_test{id}"
    #
    #     for i, f_train in enumerate(results_dir.glob("*results*.csv")):
    #         print (f_train)
    #         s = f_train.stem
    #         train_dataset = s.split("_different")[0]
    #         train_dataset_type = "fixed" if f_train.match("*fixed*") else "all"
    #
    #         df = pd.read_csv(f_train)
    #         best = df[df["rank_test_score"]==1]
    #         params = (best.head(1)["params"]).item()
    #         params =  ast.literal_eval(params)
    #
    #
    #         f_train_data = train_test_dir / f"{train_dataset}_different_train_{train_dataset_type}.csv"
    #         train = pd.read_csv(f_train_data)
    #         X_train = train.drop(["Label"], axis=1)
    #         y_train = train.Label.ravel()
    #
    #         print (f"Train file={f_train_data}")
    #         model = XGBClassifier(**params)
    #         print(model)
    #         model.fit(X_train, y_train)
    #
    #         for f_test in train_test_dir.glob("*test*"):
    #             test_dataset = str(f_test.stem).split("_test")[0]
    #             if train_dataset==test_dataset:
    #                 continue
    #             print (f"Test file={f_test}")
    #             test = pd.read_csv(f_test)
    #             X_test = test.drop(["Label"], axis=1)
    #             y_test = test.Label.ravel()
    #             score =  model.score(X_test, y_test)
    #
    #             res_table.loc[f"{train_dataset_type}_{train_dataset}", test_dataset] = round(score, 3)
    #         print(res_table)
    #     #
    #     #     # if i > 7:
    #     #     #     break
    #     res_table.sort_index(axis=0,  inplace=True)
    #     res_table.sort_index(axis=1,  inplace=True)
    #
    #     print(res_table)
    #     print(res_table.to_latex())
    #
    #     res_table.to_csv(results_dir/"summary.csv")



def self_size_summary():
    results_dir = Path("Results") / "train_test_size"
    train_test_dir = Path("Features/CSV/") / "train_test_size"

    res_table = pd.DataFrame(columns=["size", "acc"])

    for train_file in train_test_dir.glob("*train_train_*.csv"):
        ds = Dataset.create_class_from_size_file(train_file)
        size_str = str(train_file.stem).split("train_train_")[1]
        res_file = next(results_dir.glob(f"*{size_str}*"))

        df = pd.read_csv(res_file)
        best = df[df["rank_test_score"]==1]
        params = (best.head(1)["params"]).item()
        params =  ast.literal_eval(params)



        # xgboost with eval
        ###################################


        fit_params = {"eval_set": [(ds.X_val, ds.y_val)],
                      "early_stopping_rounds": 50}
        model = XGBClassifier(**params)
        model.fit(ds.X, ds.y, **fit_params)
        test = pd.read_csv(train_test_dir/ "cattle_dataset1_test.csv")

        acc =  model.score(ds.X_test, ds.y_test)
        print ("accc")
        print (acc)
        new_row = pd.DataFrame(data=[[ds.X.shape[0], round(acc, 3)]],
                                                  columns=["size", "acc"])
        res_table = res_table.append(new_row, ignore_index=True)

        print(res_table)

        # if i > 7:
        #     break
    print(res_table)
    print(res_table.to_latex())

    res_table.to_csv(results_dir/"summary.csv")




def XGBS_explore_params_self():
    results_dir = Path("Results") / "self"


    all_params = None
    for i, f in enumerate(results_dir.glob("*XGBoost*.csv")):
        print(f)
        df = pd.read_csv(f)
        best = df[df["rank_test_score"] <= 2]
        if all_params is None:
            all_params = best
        else:
            all_params = pd.concat([all_params, best], ignore_index=True)
    all_params = all_params.loc[:, all_params.columns.str.startswith('param_')]

    # print(all_params)
    # print(all_params.columns)
    for c in all_params.columns:
        print (c)
        print (pd.unique(all_params[c]))


def XGBS_explore_params_diff():
    results_dirs = [x for x in Path("Results").iterdir() if x.is_dir() and x.match("*different*")]

    all = pd.DataFrame()
    for i, f_train in enumerate(results_dirs[1].glob("*results*.csv")):
        print(f_train)
        for results_dir in results_dirs:
            s = f_train.stem
            f = next(results_dir.glob(f"*{s}*"))
            print(f)

            df = pd.read_csv(f)
            best = df[df["rank_test_score"] == 1]
            # print(best)
            all = all.append(best)
            all = all.append(pd.Series(), ignore_index=True)

        print (all)
        all.to_csv("all.csv")
        exit(3)

def xgbs_feature_importance(clf: XGBClassifier, X_train: DataFrame):
    def series2bin(d: pd.DataFrame, bins: list):
        for l in bins:
            d[f"bins{l}"] = d["rank"].apply(lambda x: int(x/l))
        return d

    feature_importances = pd.DataFrame(clf.feature_importances_, index=X_train.columns, columns=['importance'])
    feature_importances.sort_values('importance', ascending=False, inplace=True)
    feature_importances["rank"] = range(feature_importances.shape[0])
    feature_importances = series2bin(feature_importances, [10, 20, 50])
    return feature_importances
#
# def sum_feature_importance():
#     In[27]: d1 = pd.read_csv("Results/self1/feature_importance.csv", index_col=0)
#
#     result = pd.concat([d0, d1], axis=1, join='inner')
#     In[27]: a["std"] = result["rank"].std(axis=1)
# In [27]: a["rank"] = result["rank"].mean(axis=1)
# In [27]: a.sort_values(by="rank").head(20)
#


# def feature_importance_table(root_dir: Path):
#     result_dirs = [d for d in root_dir.glob("self*") if d.is_dir()]
#     for dataset in DATASET_LIST:
#         importance_files = [f for d in result_dirs for f in d.glob("*imp*")]
#         importance_df = [pd.read_csv(f, index_col=0) for f in importance_files]
#         print (len(importance_df))
#         importance_df = [df.sort_index() for df in importance_df]
#
#         join_df = pd.concat(importance_df, axis=1)
#         r = join_df["rank"]
#         res = pd.DataFrame(columns=["mean", "std"])
#         res["mean"] = r.mean(axis=1)
#         res["std"] = r.std(axis=1)
#         res.sort_values(by="mean", inplace=True)
#
#         print (res.head(20))

def feature_importance_table(NUM_OF_FEATURES:int= 15):
    output_file = Path("Results") / f"feature_importance_table.csv"
    output_file25 = Path("Results") / f"feature_importance_table25.csv"

    mstd_df_list = []
    mstd_table = pd.DataFrame()
    feature_list = []
    for dataset in DATASET_LIST:
        print(dataset)
        mstd_df, m, _ = mean_std(dir=Path("Results"), match="self*", summary_shape=(490, 5),
                                 summary_file_name=f"feature_importance_{dataset}.csv")
        mstd_df_list.append(mstd_df.sort_values(by="rank")["rank"].head(NUM_OF_FEATURES).to_latex())
        m_top = m.sort_values(by="rank").head(NUM_OF_FEATURES)
        for item in m_top.index:
            feature_list.append(item)

        mstd_table = pd.concat([mstd_table, m["rank"].rename(dataset)], axis=1, sort=False)

    mstd_table["mean"] = mstd_table.mean(axis=1)
    mstd_table.sort_values(by="mean", inplace=True, axis=0)
    mstd_table.to_csv(output_file)
    mstd_table.head(25).to_csv(output_file25)

    print (40*"*")

    print (mstd_table.astype("int").head(30).to_latex())

    print (40*"*")
    for i in range(len(mstd_df_list)):
        print ()
        print ()
        print (DATASET_LIST[i])
        print ()
        print (mstd_df_list[i])



    # print(feature_list)
    # feature_list = list(set(feature_list))
    # print(feature_list)
    # mstd_table_top = mstd_table.loc[feature_list, :]
    #

def feature_importance_correlation():
    mstd_table = pd.DataFrame()
    corr_df_dict = {"pearson": pd.DataFrame(),
                    "spearman": pd.DataFrame()}

    for dataset in DATASET_LIST:
        print(dataset)
        mstd_df, m, _ = mean_std(dir=Path("Results"), match="self*", summary_shape=(490, 5),
                                 summary_file_name=f"feature_importance_{dataset}.csv")
        mstd_table = pd.concat([mstd_table, m["importance"].rename(dataset)], axis=1, sort=False)
    mstd_table.to_csv("Results/alldatasets_feature_importance.csv")
    for d1, d2 in combinations(mstd_table.columns, 2):
        for corr_method in ["pearson", "spearman"]:
            x = mstd_table[d1]
            y = mstd_table[d2]
            corr_df_dict[corr_method].loc[d1, d2] = x.corr(y, method=corr_method, min_periods=None)

    fig = plt.figure(figsize=(16, 10))
    for index, (k,v) in enumerate(corr_df_dict.items()):
        ax = fig.add_subplot(1, 2, index+1, title=k)
        ax = heatmap(v, cmap='coolwarm', annot=True)
    plt.savefig("Results/feature_corr.pdf", format="pdf", bbox_inches='tight')




def main():
    FeatureReader.reader_selection_parameter = "without_hot_encoding"

    ##########################
    # self summary
    ##########################
    # p = Pool(20)
    # p.map(self_results_summary, range(20))

    #########################
    # collect the self summary information
    #########################
    # mean_std(Path("Results"), "self*", (8, 6))
    # exit (7)

    ##########################
    # xgbs measurement
    ##########################
    # mean_std(Path("Results"), "self*", (8, 8), summary_file_name="xgbs_measurements.csv")
    # exit(7)
    ##########################
    # compare self to random splits
    ##########################
    # p = Pool(10)
    # self_results_summary_random = partial(self_results_summary, prefix="random_")
    # p.map(self_results_summary_random, range(5))
    #
    # mean_std(Path("Results"), "random_self*", (8, 1))
    #
    # exit(7)

    ##########################
    # Xgboost feature importance
    ##########################

    # feature_importance_table()
    feature_importance_correlation()
    exit(5)

    ##########################
    # different_results_summary
    ##########################
    # p = Pool(20)
    # p.map(different_results_summary, range(20))
    # exit(7)
    _, m, _ = mean_std(dir=Path("Results"), match="self*", summary_shape=(8, 8), summary_file_name="diff_summary.csv")
    m.to_csv(Path("Results")/"diff_summary_mean.csv")

    exit(7)




    # ['human_dataset1', 'mouse_dataset1', 'celegans_dataset2', 'celegans_dataset1', 'human_dataset2', 'human_dataset3',
     # 'cattle_dataset1', 'mouse_dataset2']
    #
    # d = Path("Results")
    # feature_importance_table(d)
    mstd_df_list = []
    mstd_table = pd.DataFrame()
    NUM_OF_FEATURES = 15
    feature_list = []
    for dataset in DATASET_LIST:
        print (dataset)
        mstd_df, m, _ = mean_std(dir=Path("Results"), match="self*", summary_shape=(580, 5),
                    summary_file_name=f"feature_importance_{dataset}.csv")
        mstd_df_list.append(mstd_df.sort_values(by="rank").head(NUM_OF_FEATURES).to_latex())
        m_top = m.sort_values(by="rank").head(NUM_OF_FEATURES)
        for item in m_top.index:
            feature_list.append(item)

        mstd_table = pd.concat([mstd_table, m["rank"]], axis=1, sort=False)



    print (feature_list)
    feature_list = list(set(feature_list))
    print (feature_list)
    print (40*"*")
    for i in range(len(mstd_df_list)):
        print ()
        print ()
        print (DATASET_LIST[i])
        print ()
        print (mstd_df_list[i])







    exit(3)


# r = []
    # for i in range (20):
    #     r.append(self_results_summary(i))
    # for t in r:
    #     print (t)



    # process_list = []
    # for i in range(10):
    #     p = Process(target=self_results_summary, args=(i, ))
    #     p.start()
    #
    #     process_list.append(p)
    # for p in process_list:
    #     p.join()

    # self_results_summary(3)
    # self_results_summary()
    # XGBS_explore_params()
    # different_results_summary()
    # different_mean_std()
    # self_size_summary()
    # XGBS_explore_params_diff()





# summary = pd.DataFrame(columns=["Organism", "Paper", "CLF", "K", "Full dataset","Val score"])
    # for f_result in results_dir.glob('*k=*'):
    #     df = pd.read_csv(f_result)
    #     print(f_result)
    #     df.sort_values(by="rank_test_score", inplace=True)
    #     df.to_csv(f_result)
    #     best_result = df.head(1)
    #     rank = best_result["rank_test_score"].item()
    #     assert rank==1, "This should be the best score"
    #     val_score = best_result["mean_test_score"].item()
    #     a = str(f_result).split("_")
    #     b = 0 if str(f_result).find("train_all")==-1 else 1
    #     k = a[1]
    #     org = a[3+b]
    #     data_ix = [i for i,x in enumerate(a) if x=="Data"][0]
    #     paper = " ".join(a[4+b:data_ix])
    #     full = b
    #     clf = [x for x in a if x.find("csv")!=-1][0].split(".csv")[0]
    #     summary = summary.append(pd.Series([org, paper, clf, k, full, val_score], index=summary.columns), ignore_index=True)
    #
    #
    #
    #     print (summary)
    # summary.to_csv(results_dir/ "summary.csv")
    #
    #



if __name__ == "__main__":
    main()




