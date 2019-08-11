import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import shuffle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Process


def neg_file (pos_file):
    dataset = str(pos_file.stem).split("pos_")[1]
    return pos_file.parent / f"neg_{dataset}.csv"

def trainAll_trainfixed_test_split (df, test_size, seed):
    ###################################
    # Select the test
    ###################################
    self_train_all, test = train_test_split(df, test_size=test_size, random_state=seed+1)

    ###################################
    # Select the train
    # Case1: train and test on the same data set
    ###################################

    try:
        self_train_fixed, tmp = train_test_split(self_train_all, train_size=test_size, random_state=seed+2)
    except ValueError:
        self_train_all   = pd.DataFrame(columns=df.columns)
        self_train_fixed = pd.DataFrame(columns=df.columns)

    ###################################
    # Select the train
    # Case2: train and test on different data set
    ###################################
    if df.shape[0]<test_size:
        different_train_all   = pd.DataFrame(columns=df.columns)
        different_train_fixed = pd.DataFrame(columns=df.columns)
    else:
        different_train_all = df.copy()
        different_train_fixed, tmp = train_test_split(different_train_all, train_size=test_size, random_state=seed+3)

    ###################################
    # Function output
    ###################################
    return test, \
           self_train_fixed, self_train_all, \
           different_train_fixed, different_train_all



def select_feature (df, first_feature="Seed_match_compact_A"):
    feature_loc = df.columns.get_loc(first_feature)
    return df[df.columns[feature_loc:]]


def create_train_test (input_dir, output_dir, minimun_pos_samples, seed):
    pos_files = list(input_dir.glob('*pos*.csv'))

    for f in pos_files:
        print (f"working on files: pos={f}\t neg={neg_file(f)}")
        pos = pd.read_csv(f)
        if pos.shape[0] < minimun_pos_samples:
            continue
        neg = pd.read_csv(neg_file(f))
        pos_lite = select_feature(pos)
        neg_lite = select_feature(neg)
        pos_lite.insert(0, "Label", 1)
        neg_lite.insert(0, "Label", 0)
        print (pos_lite.shape)

        pos_test, \
        pos_self_train_fixed, pos_self_train_all, \
        pos_different_train_fixed, pos_different_train_all = \
            trainAll_trainfixed_test_split(df=pos_lite, test_size=minimun_pos_samples, seed=seed)

        neg_test, \
        neg_self_train_fixed, neg_self_train_all, \
        neg_different_train_fixed, neg_different_train_all = \
            trainAll_trainfixed_test_split(df=neg_lite, test_size=minimun_pos_samples, seed=seed*100)


        test                    = pd.concat([pos_test, neg_test], ignore_index=True)
        self_train_fixed        = pd.concat([pos_self_train_fixed, neg_self_train_fixed], ignore_index=True)
        self_train_all          = pd.concat([pos_self_train_all, neg_self_train_all], ignore_index=True)
        different_train_fixed   = pd.concat([pos_different_train_fixed, neg_different_train_fixed], ignore_index=True)
        different_train_all     = pd.concat([pos_different_train_all, neg_different_train_all], ignore_index=True)

        var_file_list = [
            (test, "test"),
            (self_train_fixed, "self_train_fixed"),
            (self_train_all, "self_train_all"),
            (different_train_fixed, "different_train_fixed"),
            (different_train_all, "different_train_all")
        ]


        dataset  = str(f.stem).split("pos_")[1]
        for  d, name  in var_file_list:
            out_file = f"{dataset}_{name}.csv"
            d = d.reindex(np.random.RandomState(seed=seed).permutation(d.index))
            d.to_csv(output_dir / out_file, index=False)




def main():
    csv_dir = Path("Features/CSV")
    process_list = []

    for i in range (1,21):
        train_test_dir = csv_dir / f"train_test{i}"
        train_test_dir.mkdir()

        ###########################################3
        # Create train_test
        ###########################################3
        p = Process(target=create_train_test, args=(csv_dir, train_test_dir, 750, i*7))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()




if __name__ == "__main__":
    main()
