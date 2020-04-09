from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns


from MirBaseUtils.mirBaseUtils import insert_mirna

csv_dir = Path("Features/CSV/")
stat_dir = csv_dir / "stat"
mir_dict_file = stat_dir / "mir_dict.pkl"
dataset_translation_list = {'human_dataset1' : "h1",'human_dataset2' : "h2",'human_dataset3' : "h3",
                            'mouse_dataset1' : "m1",'mouse_dataset2' : "m2",
                            'celegans_dataset1' : "ce1", 'celegans_dataset2' : "ce2",
                            'cattle_dataset1' : "ca1"}
import numpy as np


def KL(P, Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P + epsilon
    Q = Q + epsilon

    divergence = np.sum(P * np.log(P / Q))
    return divergence



def dataset_statistics():
    stat_dir.mkdir(exist_ok=True)

    for dataset_file in csv_dir.glob("*positive_feature*"):
        print(dataset_file)
        dataset  = str(dataset_file.stem).split("_duplex")[0]
        pos_dataset = pd.read_csv(dataset_file)
        cnt = pos_dataset["microRNA_name"].value_counts()
        stat = pd.DataFrame()
        cnt.name = "All"
        stat = pd.concat([stat, cnt], axis=1, sort=False)
        # for split in dir.glob(f"*{dataset}*"):
        #     if split.match("*stat*"):
        #         continue
        #     print (split)
        #     split_name = str(split.stem).split(dataset)[1][1:]
        #     df = pd.read_csv(split)
        #     df = df[df["Label"]==1]
        #     cnt = df["microRNA_name"].value_counts()
        #     cnt.name = split_name
        #     stat = pd.concat([stat, cnt], axis=1, sort=False)
        # stat = stat.fillna(0)
        stat = stat.astype(int)
        stat = stat[stat.columns.sort_values()]

        stat.to_csv(stat_dir/f"{dataset}_stat.csv")
        print (stat.head(3))



def dataset_statistics_plot():


    f = plt.figure()
    ax = plt.subplot(1, 1, 1)

    res = pd.DataFrame()
    stat_files = list(stat_dir.glob(f"*stat*"))
    top_mir_dict = {}
    for k, stat_f in enumerate(stat_files):
        print(stat_f)
        label = stat_f.stem.replace("_stat", "")
        stat = pd.read_csv(stat_f, index_col=0)
        stat.sort_values(by='All', ascending=False, inplace=True)
        v = stat.cumsum()
        ninety_flag = v <= v.iloc[-1] * 0.9
        mir_list = list((ninety_flag.loc[ninety_flag['All']]).index)
        top_mir_dict[label] = mir_list

        v = stat['All']
        res.loc["Dataset size", label] = v.cumsum().iloc[-1]
        res.loc["# miRNAs", label] = len(v)
        ninety = (v.cumsum() <= v.cumsum().iloc[-1] * 0.9).idxmin()
        ninety = v.index.get_loc(ninety)
        assert ninety==len(mir_list), f"wrong calculation of ninety: {ninety}!={len(mir_list)}"
        res.loc["# miRNAs describes 90% dataset", label] = f"{ninety} ({int(ninety/len(v)*100)}%)"
        # res.loc["% miRNAs describes 90% dataset", label] = ninety/len(v)*100
        # mark = [vals.index(i) for i in roots]
        # print(mark)
        # plt.plot(vals, poly, markevery=mark, ls="", marker="o", label="points")

        vv = v.cumsum()
        vv.index = range(1, len(vv)+1)
        vv.plot(label=dataset_translation_list[label], markevery=[ninety], marker="o")
        plt.xlabel('No. of miRNA sequences')
        plt.ylabel('No. of interactions')

    with mir_dict_file.open("wb") as f:
        pickle.dump(top_mir_dict, f)


    handles, labels = ax.get_legend_handles_labels()
    import operator
    hl = sorted(zip(handles, labels),
                key=operator.itemgetter(1))
    handles2, labels2 = zip(*hl)

    ax.legend(handles2, labels2)
    res = res.reindex(sorted(res.columns), axis=1)


    plt.savefig(stat_dir/f"mirna_dist.pdf", format="pdf", bbox_inches='tight')
    # res = res.astype(int)
    print(res.to_latex())

# def mir_conservation_stat():
#     with mir_dict_file.open("rb") as f:
#         top_mir_dict = pickle.load(f)
#     mir_df_dict = {}
#     col_name = "mir_name"
#     for dataset, mir_list in top_mir_dict.items():
#         df = pd.DataFrame({col_name : mir_list})
#         df = insert_mirna(df, col_name)
#         df["seed"] = df["miRNA_seq"].apply(lambda x: x[1:7])
#         # print (df)
#         seed_list = df["seed"].tolist()
#         # print(len(seed_list))
#         seed_set = set(seed_list)
#         # print(len(seed_set))
#         mir_df_dict[dataset] = df
#     res_table = pd.DataFrame()
#     for source_name, source_df in mir_df_dict.items():
#         for target_name, target_df in mir_df_dict.items():
#             print(target_name)
#             target_list = target_df["seed"].tolist()
#             source_df["match"]=source_df["seed"].apply(lambda x: x in target_list)
#             res_table.loc[target_name, source_name] = sum(source_df["match"])/source_df.shape[0]
#             if source_name=="human_dataset1" and target_name=="cattle_dataset1":
#                 print("hello")
#
#     res_table.to_csv(stat_dir/"mir_conservation.csv")
#     exit(3)

def mir_conservation_stat():
    def get_seed(x: str)->str:
        x = x.upper().replace('T', 'U')
        return x[1:7]

    dataset_files = list(Path("Datafiles_Prepare/CSV").glob("*duplex_positive.csv"))

    mir_weight_cache = {}
    for f in dataset_files:
        df = pd.read_csv(f, usecols=["miRNA sequence"])
        df["seed"] = df["miRNA sequence"].apply(get_seed)
        mir_weight_cache[f] = df["seed"].value_counts(normalize=True)

    seed_stat_df: pd.DataFrame = pd.DataFrame()
    for f, vc in mir_weight_cache.items():
        vc.sort_values(ascending=False, inplace=True)
        dataset = f.stem.replace("duplex_positive_", "")
        seed_stat_df.loc["seed_count", dataset] = len(vc)
        ninety = (vc.cumsum() <= vc.cumsum().iloc[-1] * 0.9).idxmin()
        ninety = vc.index.get_loc(ninety)
        seed_stat_df.loc["ninety_percent", dataset] = ninety
        print (seed_stat_df)
    seed_stat_df.sort_index(axis=1, inplace=True)
    seed_stat_df = seed_stat_df.astype("int")
    print(seed_stat_df.to_latex())



    res_table = pd.DataFrame()
    for test_file in dataset_files:
        test_name = test_file.stem.replace("_duplex_positive", "")
        test_seed_weight = mir_weight_cache[test_file]
        for train_file in dataset_files:
            train_name = train_file.stem.replace("_duplex_positive", "")
            train_seed_weight = mir_weight_cache[train_file]
            print(train_name)
            join_result = pd.merge(test_seed_weight.to_frame(), train_seed_weight.to_frame(), how='outer', left_index=True, right_index=True)
            join_result.fillna(0, inplace=True)
            p = join_result.ix[:,0].values
            q = join_result.ix[:,1].values


            # test_df["score"] = test_df["seed"].apply(lambda x: get_weight(train_seed_weight, x))
            res_table.loc[train_name, test_name] = KL(p, q)
            print(res_table)
            # if source_name == "human_dataset1" and train_name == "cattle_dataset1":
            #     print("hello")

    res_table.to_csv(stat_dir / "mir_conservation.csv")
    print(res_table)

    exit(3)
    #
    # def mir_conservation_stat():
    #     def get_seed(x: str) -> str:
    #         x = x.upper().replace('T', 'U')
    #         return x[1:7]
    #
    #     def get_weight(weights: pd.Series, seed: str) -> float:
    #         if seed not in weights.index:
    #             return 0
    #         return weights.get(seed)
    #
    #     dataset_files = list(Path("Datafiles_Prepare/CSV").glob("*duplex_positive.csv"))
    #
    #     mir_weight_cache = {}
    #     for f in dataset_files:
    #         df = pd.read_csv(f, usecols=["miRNA sequence"])
    #         df["seed"] = df["miRNA sequence"].apply(get_seed)
    #         mir_weight_cache[f] = df["seed"].value_counts(normalize=True)
    #
    #     res_table = pd.DataFrame()
    #     for test_file in dataset_files:
    #         test_df = pd.read_csv(test_file, usecols=["miRNA sequence"])
    #         test_name = test_file.stem.replace("_duplex_positive", "")
    #         test_df["seed"] = test_df["miRNA sequence"].apply(get_seed)
    #         for train_file in dataset_files:
    #             train_name = train_file.stem.replace("_duplex_positive", "")
    #             print(train_name)
    #             train_seed_weight = mir_weight_cache[train_file]
    #             test_df["score"] = test_df["seed"].apply(lambda x: get_weight(train_seed_weight, x))
    #             res_table.loc[train_name, test_name] = sum(test_df["score"]) / test_df.shape[0]
    #             print(res_table)
    #             # if source_name == "human_dataset1" and train_name == "cattle_dataset1":
    #             #     print("hello")
    #
    #     res_table.to_csv(stat_dir / "mir_conservation.csv")
    #     print(res_table)
    #
    #     exit(3)

    # def mir_conservation_stat():
    #     def get_seed(x: str) -> str:
    #         x = x.upper().replace('T', 'U')
    #         return x[1:7]
    #
    #     dataset_files = list(Path("Datafiles_Prepare/CSV").glob("*duplex_positive.csv"))
    #
    #     res_table = pd.DataFrame()
    #     for source_file in dataset_files:
    #         source_df = pd.read_csv(source_file, usecols=["miRNA sequence"])
    #         source_name = source_file.stem.replace("_duplex_positive", "")
    #         source_df["seed"] = source_df["miRNA sequence"].apply(get_seed)
    #         for target_file in dataset_files:
    #             target_df = pd.read_csv(target_file, usecols=["miRNA sequence"])
    #             target_name = target_file.stem.replace("_duplex_positive", "")
    #             print(target_name)
    #             target_df["seed"] = target_df["miRNA sequence"].apply(get_seed)
    #             target_list = target_df["seed"].tolist()
    #             source_df["match"] = source_df["seed"].apply(lambda x: x in target_list)
    #             res_table.loc[target_name, source_name] = sum(source_df["match"]) / source_df.shape[0]
    #             # if source_name == "human_dataset1" and target_name == "cattle_dataset1":
    #             #     print("hello")
    #
    #     res_table.to_csv(stat_dir / "mir_conservation.csv")
    #     exit(3)

    #
    # def insert_mirna(in_df, col_name, organism_prefix=None):
    #     def find_mrna(row):
    #         mirna_id = row[col_name]
    #         try:
    #             return miRBase_df[miRBase_df["mi_name"] == mirna_id]["mi_seq"].values[0]
    #         except IndexError:
    #             return "No mrna match!!!"
    #
    #     miRBase_df = read_mirbase_file(organism_prefix)
    #     in_df['miRNA_seq'] = in_df.apply(find_mrna, axis=1)
    #     return in_df.copy()
    #
    #
    #


def plot_mir_pair_dist():
    def get_seed(x: str) -> str:
        x = x.upper().replace('T', 'U')
        return x[1:7]

    dataset_files = list(Path("Datafiles_Prepare/CSV").glob("*duplex_positive.csv"))

    mir_weight_cache = {}
    for f in dataset_files:
        df = pd.read_csv(f, usecols=["miRNA sequence"])
        df["seed"] = df["miRNA sequence"].apply(get_seed)
        mir_weight_cache[f] = df["seed"].value_counts(normalize=True)

    join_table = pd.DataFrame()
    for file in dataset_files:
        name = file.stem.replace("_duplex_positive", "")
        seed_weight = mir_weight_cache[file]
        seed_weight.rename(name, inplace=True)
        join_table = pd.merge(join_table, seed_weight.to_frame(), how='outer',
                                   left_index=True, right_index=True)
        join_table.fillna(0, inplace=True)
    join_table.to_csv(stat_dir/ "seed_across_all_dataset.csv")
    exit(3)

    # for c1 in join_table.columns:
    #     for c2 in join_table.columns:
    #         if c1==c2:
    #             continue
    #         print (f"seed_{c1}_{c2}")
    #         plt.figure()
    #         d = join_table[[c1, c2]]
    #         d = d[(d[c1]!=0) | (d[c2]!=0)]
    #         d.sort_values(by=c1, ascending=False, inplace=True)
    #         plt.plot(d.index.astype("str"), d[c1], ".")
    #         plt.plot(d.index.astype("str"), d[c2], ".")
    #
    #         plt.savefig(stat_dir / f"seed_{c1}_{c2}.pdf", format="pdf", bbox_inches='tight')
    #
    #         plt.figure()
    #         sns.scatterplot(
    #             x=c1, y=c2,
    #             palette=sns.color_palette("hls", 10),
    #             data=d,
    #             legend=False,
    #             alpha=0.3
    #         )
    #         # Set x-axis label
    #         plt.xlabel('')
    #         # Set y-axis label
    #         plt.ylabel('')
    #
    #         plt.savefig(stat_dir / f"seed_scatter_{c1}_{c2}.pdf", format="pdf", bbox_inches='tight')
    #         print("scatter")
    #

def main():
    # dataset_statistics()
    dataset_statistics_plot()
    # mir_conservation_stat()
    # plot_mir_pair_dist()




if __name__ == "__main__":
    main()

