
import pandas as pd
from pathlib import Path
from collections import ChainMap
from TPVOD_Utils import utils
import json
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from xgboost import XGBClassifier

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


def self_results_summary():
    results_dir = Path("Results") / "self"
    train_test_dir = Path("Features/CSV/train_test0")

    res_table = pd.DataFrame(index=pd.MultiIndex.from_tuples([], names=['Dataset', 'Train']))

    for f in train_test_dir.glob("*self_train_all*"):
        if f.stat().st_size < 99999:
            continue
        dataset = str(f.stem).split("_self")[0]
        self_train_fixed        = f"{dataset}_self_train_fixed.csv"
        self_train_all          = f"{dataset}_self_train_all.csv"
        res_table.loc[(dataset, "fixed"), "train size"] = pd.read_csv(train_test_dir/self_train_fixed).shape[0]
        res_table.loc[(dataset, "all"), "train size"] = pd.read_csv(train_test_dir/self_train_all).shape[0]

    print(res_table)


    for i, f in enumerate(results_dir.glob("*results*.csv")):
        print (f)
        s = f.stem
        dataset = s.split("_self")[0]
        method = s.split("_results_")[1]
        train = s.split("_self_")[1]
        train = train.split("_results_")[0]
        # print (dataset)
        # print(method)
        # print(train)
        df = pd.read_csv(f)
        best = df[df["rank_test_score"]==1]
        params = (best.head(1)["params"]).item()
        params =  ast.literal_eval(params)
        # print (params)
        # index = pd.MultiIndex.from_tuples([(dataset, train)], names=['Dataset', 'Train'])

        # d = f"{dataset}_{train}"
        res_table.loc[(dataset, train), method] = round(run_test(dataset, train, params, method), 3)

        print(res_table)

        # if i > 7:
        #     break
    res_table.sort_values(by='Dataset', inplace=True)
    print(res_table)
    print(res_table.to_latex())

    res_table.to_csv(results_dir/"summary.csv")



def different_results_summary():
    results_dir = Path("Results") / "different"
    train_test_dir = Path("Features/CSV/train_test")

    res_table = pd.DataFrame()



    for i, f_train in enumerate(results_dir.glob("*results*.csv")):
        print (f_train)
        s = f_train.stem
        train_dataset = s.split("_different")[0]

        df = pd.read_csv(f_train)
        best = df[df["rank_test_score"]==1]
        params = (best.head(1)["params"]).item()
        params =  ast.literal_eval(params)


        f_train_data = train_test_dir / f"{train_dataset}_different_train_fixed.csv"
        train = pd.read_csv(f_train_data)
        X_train = train.drop(["Label"], axis=1)
        y_train = train.Label.ravel()


        model = XGBClassifier(**params)
        print(model)
        model.fit(X_train, y_train)

        for f_test in train_test_dir.glob("*test*"):
            test_dataset = str(f_test.stem).split("_test")[0]
            if train_dataset==test_dataset:
                continue
            test = pd.read_csv(f_test)
            X_test = test.drop(["Label"], axis=1)
            y_test = test.Label.ravel()
            score =  model.score(X_test, y_test)

            res_table.loc[train_dataset, test_dataset] = round(score, 3)
            print(res_table)
    #
    #     # if i > 7:
    #     #     break
    res_table.sort_index(axis=0,  inplace=True)
    res_table.sort_index(axis=1,  inplace=True)

    print(res_table)
    print(res_table.to_latex())

    res_table.to_csv(results_dir/"summary.csv")


def XGBS_explore_params():
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


def main():
    self_results_summary()
    # XGBS_explore_params()
    # different_results_summary()





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




