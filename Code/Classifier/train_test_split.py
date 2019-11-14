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
import click


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
    col_to_save = list(df.columns[feature_loc:])
    col_to_save.insert(0, "microRNA_name")

    return df[col_to_save]


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


def stratify_train_test_split (df, test_size , random_state ):
    #Change: all the unique miRNA were put in the test set
    uniques_mirna = df[df.groupby("microRNA_name").microRNA_name.transform(len)==1]
    non_uniques_mirna = df[df.groupby("microRNA_name").microRNA_name.transform(len)>1]

    #dealing with the non_uniques_mirna
    non_uniques_train, non_uniques_test = train_test_split(non_uniques_mirna, test_size =test_size ,
                                                           random_state=random_state,
                                                           stratify= non_uniques_mirna["microRNA_name"])

    train = pd.concat([non_uniques_train])
    test = pd.concat([non_uniques_test, uniques_mirna])
    return train, test



def create_stratify_train_test_split (input_dir, output_dir, test_size, seed):
    def train_val_test_split (df, label):
        df_dict = {}
        df_dict["train"], df_dict["test"] = stratify_train_test_split(df, test_size, seed)
        df_dict["train_train"], df_dict["train_val"] = stratify_train_test_split(df_dict["train"], test_size,
                                                                                       seed)
        for key in df_dict.keys():
            df_dict[key] = select_feature(df_dict[key])
            df_dict[key].insert(0, "Label", label)

        return df_dict

    pos_files = list(input_dir.glob('*pos*.csv'))

    for f in pos_files:
        print (f"working on files: pos={f}\t neg={neg_file(f)}")
        #Create the positives
        pos = pd.read_csv(f)
        pos_dict = train_val_test_split(pos, 1)

        # Create the negatives
        neg = pd.read_csv(neg_file(f))
        neg_dict = train_val_test_split(neg, 0)

        # concat the pos & neg
        for key in pos_dict.keys():
            pos_dict[key] = pos_dict[key].append(neg_dict[key], ignore_index=True)

        # save to csv
        dataset  = str(f.stem).split("pos_")[1]
        for key, d in pos_dict.items():
            out_file = f"{dataset}_{key}.csv"
            d = d.reindex(np.random.RandomState(seed=seed).permutation(d.index))
            d.to_csv(output_dir / out_file, index=False)


def create_train_dataset_in_steps (f:Path, step_size: int, random_state: int):
    print (f"step_size: {step_size}")
    d = pd.read_csv(f)
    d_size = d.shape[0]
    for i in range(int(d_size/step_size)):
        pos = d[d["Label"]==1]
        neg = d[d["Label"]==0]
        test_size = 1 - (d_size-step_size)/float(d_size)
        print (f"test_size = {test_size}")
        pos_train, _ = stratify_train_test_split(pos, test_size, random_state)
        neg_train, _ = stratify_train_test_split(neg, test_size, random_state)
        train = pos_train.append(neg_train, ignore_index=True)
        train = train.reindex(np.random.RandomState(seed=random_state).permutation(train.index))
        out_file = f.parent/ f"{f.stem}_{d_size}.csv"
        train.to_csv (out_file)
        print (train.shape)

        d = train
        d_size = d.shape[0]


def dataset_statistics(id):
    csv_dir =  Path("Features/CSV/")
    dir = csv_dir / f"train_test{id}"
    print (dir)
    for dataset_file in dir.glob("*test*"):
        dataset  = str(dataset_file.stem).split("_test")[0]
        pos_dataset = pd.read_csv(csv_dir/f"pos_{dataset}.csv")
        cnt = pos_dataset["microRNA_name"].value_counts()
        stat = pd.DataFrame()
        cnt.name = "All"
        stat = pd.concat([stat, cnt], axis=1, sort=False)
        for split in dir.glob(f"*{dataset}*"):
            if split.match("*stat*"):
                continue
            print (split)
            split_name = str(split.stem).split(dataset)[1][1:]
            df = pd.read_csv(split)
            df = df[df["Label"]==1]
            cnt = df["microRNA_name"].value_counts()
            cnt.name = split_name
            stat = pd.concat([stat, cnt], axis=1, sort=False)
        stat = stat.fillna(0)
        stat = stat.astype(int)
        stat = stat[stat.columns.sort_values()]

        stat.to_csv(dir/f"stat_{dataset}.csv")
        print (stat.head(3))

def train_test_size(id=0):
    csv_dir =  Path("Features/CSV/")
    dir = csv_dir / f"train_test{id}"
    print (dir)
    result = pd.DataFrame()
    for dataset_file in dir.glob("*test*"):
        dataset  = str(dataset_file.stem).split("_test")[0]
        for suffix in ["train_train", "train_val", "test"]:
            d = pd.read_csv(dir / f"{dataset}_{suffix}.csv")
            suffix = suffix.replace("train_", "")
            result.loc[dataset, suffix] = int(d.shape[0])
    result.sort_index(inplace=True)
    result = result.astype('int')
    print(result)
    print(60 * "*")
    print(result.to_latex())


def dataset_statistics_plot(id=0):
    csv_dir = Path("Features/CSV/")
    dir = csv_dir / f"train_test{id}"

    f = plt.figure()
    res = pd.DataFrame()
    stat_files = list(dir.glob(f"*stat*"))
    for k, stat_f in enumerate(stat_files):
        stat = pd.read_csv(stat_f)
        v = stat["All"]
        label = str(stat_f.stem).split("stat_")[1]
        label = label.replace("_dataset", "")
        v.cumsum().plot(label=label)
        plt.xlabel('# miRNA')
        plt.ylabel('# Samples')

        res.loc["Dataset size", label] = v.cumsum().iloc[-1]
        res.loc["# miRNAs", label] = len(v)
        ninety = (v.cumsum() > v.cumsum().iloc[-1] * 0.9).argmax()
        res.loc["# miRNAs describes 90% dataset", label] = ninety
        res.loc["% miRNAs describes 90% dataset", label] = ninety/len(v)*100


    L = plt.legend()

    plt.legend()

    plt.savefig(dir/f"dataset_plot.pdf", format="pdf", bbox_inches='tight')
    res = res.astype(int)
    print(res.to_latex())


@click.command()
@click.option('--split',  default=0, type=int, help="split the datasets to train, test, val. you ave to provide how many time to do this")
@click.option('--stat', is_flag=True,  help='Generate stat')
@click.option('--split_size',  default=(None, 0, 1), type=(Path, int, int), help="split the train into smaller and smaller parts")

def cli(split,stat, split_size):
    csv_dir = Path("Features/CSV")
    process_list = []

    for i in range (split):
        train_test_dir = csv_dir / f"train_test{i}"
        train_test_dir.mkdir()

        ###########################################3
        # Create train_test
        ###########################################3
        p = Process(target=create_stratify_train_test_split, args=(csv_dir, train_test_dir, 0.2, i*7))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()

    if stat:
        # process_list = []
        # for i in range(10):
        #     p = Process(target=dataset_statistics, args=(i, ))
        #     p.start()
        #     process_list.append(p)
        # for p in process_list:
        #     p.join()

        dataset_statistics_plot()
    f, step_size, random_state = split_size
    if f is not None:
        create_train_dataset_in_steps(Path(f), step_size, random_state)


def main():
    # dataset_statistics()
    # cli ()
    train_test_size()
#
# def main():
#     csv_dir = Path("Features/CSV")
#     process_list = []
#
#     for i in range (10):
#         train_test_dir = csv_dir / f"train_test{i}"
#         train_test_dir.mkdir()
#
#         ###########################################3
#         # Create train_test
#         ###########################################3
#         p = Process(target=create_stratify_train_test_split, args=(csv_dir, train_test_dir, 0.2, i*7))
#         p.start()
#         process_list.append(p)
#     for p in process_list:
#         p.join()
#
#


if __name__ == "__main__":
    main()
