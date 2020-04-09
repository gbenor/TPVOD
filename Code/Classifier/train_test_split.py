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

@click.group()
def cli():
    pass



def neg_file (pos_file):
    n = str(pos_file)
    n = n.replace("positive", "negative")
    return Path(n)


def stratify_train_test_split (df, test_size , random_state):
    #Change: all the unique miRNA were put in the test set
    uniques_mirna = df[df.groupby("microRNA_name").microRNA_name.transform(len)==1]
    non_uniques_mirna = df[df.groupby("microRNA_name").microRNA_name.transform(len)>1]

    #dealing with the non_uniques_mirna
    non_uniques_train, non_uniques_test = train_test_split(non_uniques_mirna, test_size=test_size,
                                                           random_state=random_state,
                                                           stratify=non_uniques_mirna["microRNA_name"])

    train = pd.concat([non_uniques_train])
    test = pd.concat([non_uniques_test, uniques_mirna])
    return train, test


def stratify_train_test_split_worker(input_file, output_dir, test_size, random_state):
    print (f"working on file: pos={input_file}")

    pos = pd.read_csv(input_file)
    pos.insert(0, "Label", 1)
    neg = pd.read_csv(neg_file(input_file))
    neg.insert(0, "Label", 0)
    #Both dataset must have the same columns
    col = [c for c in pos.columns if c in neg.columns]
    pos = pos[col]
    neg = neg[col]

    pos_train, pos_test = stratify_train_test_split(pos, test_size, random_state)
    neg_train, neg_test = stratify_train_test_split(neg, test_size, random_state)

    # concat the pos & neg
    output = {
        "train" : pos_train.append(neg_train, ignore_index=True),
        "test" : pos_test.append(neg_test, ignore_index=True)
    }

    # save to csv
    dataset  = str(input_file.stem).split("_duplex_")[0]
    for key, d in output.items():
        out_file = f"{dataset}_{key}.csv"
        d = d.reindex(np.random.RandomState(seed=random_state).permutation(d.index))
        d.to_csv(output_dir / out_file, index=False)


def random_train_test_split_worker(input_file, output_dir, test_size, random_state):
    print (f"working on file: pos={input_file}")

    pos = pd.read_csv(input_file)
    pos.insert(0, "Label", 1)
    neg = pd.read_csv(neg_file(input_file))
    neg.insert(0, "Label", 0)
    #Both dataset must have the same columns
    col = [c for c in pos.columns if c in neg.columns]
    pos = pos[col]
    neg = neg[col]

    pos_train, pos_test = train_test_split(pos, test_size=test_size, random_state=random_state)
    neg_train, neg_test = train_test_split(neg, test_size=test_size, random_state=random_state)

    # concat the pos & neg
    output = {
        "train" : pos_train.append(neg_train, ignore_index=True),
        "test" : pos_test.append(neg_test, ignore_index=True)
    }

    # save to csv
    dataset  = str(input_file.stem).split("_duplex_")[0]
    for key, d in output.items():
        out_file = f"{dataset}_{key}.csv"
        d = d.reindex(np.random.RandomState(seed=random_state).permutation(d.index))
        d.to_csv(output_dir / out_file, index=False)





@cli.command()
@click.option('--n', type=int, required=True,
              help="How many time to do this")
def split(n):
    test_size = 0.2
    csv_dir = Path("Features/CSV")
    process_list = []

    for i in range (n):
        train_test_dir = csv_dir / f"train_test{i}"
        train_test_dir.mkdir(exist_ok=True)

        for f in csv_dir.glob("*duplex_positive_feature*.csv"):
            ###########################################3
            # Create train_test
            ###########################################3
            p = Process(target=stratify_train_test_split_worker, args=(f, train_test_dir, test_size, i*19))
            p.start()
            process_list.append(p)
    for p in process_list:
        p.join()

@cli.command()
@click.option('--n', type=int, required=True,
              help="How many time to do this")
def split_random(n):
    test_size = 0.2
    csv_dir = Path("Features/CSV")
    process_list = []

    for i in range (n):
        train_test_dir = csv_dir / f"random_train_test{i}"
        train_test_dir.mkdir(exist_ok=True)

        for f in csv_dir.glob("*duplex_positive_feature*.csv"):
            ###########################################3
            # Create train_test
            ###########################################3
            p = Process(target=random_train_test_split_worker, args=(f, train_test_dir, test_size, i*7))
            p.start()
            process_list.append(p)
    for p in process_list:
        p.join()





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

#
# @click.command()
# @click.option('--split',  default=0, type=int, help="split the datasets to train, test, val. you ave to provide how many time to do this")
# @click.option('--stat', is_flag=True,  help='Generate stat')
# @click.option('--split_size',  default=(None, 0, 1), type=(Path, int, int), help="split the train into smaller and smaller parts")
#
# def cli(split,stat, split_size):
#     csv_dir = Path("Features/CSV")
#     process_list = []
#
#     for i in range (split):
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
#     if stat:
#         # process_list = []
#         # for i in range(10):
#         #     p = Process(target=dataset_statistics, args=(i, ))
#         #     p.start()
#         #     process_list.append(p)
#         # for p in process_list:
#         #     p.join()
#
#         dataset_statistics_plot()
#     f, step_size, random_state = split_size
#     if f is not None:
#         create_train_dataset_in_steps(Path(f), step_size, random_state)
#




if __name__ == "__main__":
    cli()
