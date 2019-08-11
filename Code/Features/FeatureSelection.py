import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def neg_file (pos_file, neg_files):
    f_suff = pos_file.stem.split("pos_")[1]
    for x in neg_files:
        if x.match(f"*{f_suff}*"):
            return x
    return None

def trainAll_trainfixed_test_split (df, test_size=750):
    train_all, test = train_test_split(df, test_size=test_size, random_state=42)
    try:
        train_fixed, tmp = train_test_split(train_all, train_size=test_size, random_state=42)
        return test, train_all, train_fixed
    except ValueError:
        return test, pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)



def select_feature (df, first_feature="Seed_match_compact_A"):
    feature_loc = df.columns.get_loc(first_feature)
    return df[df.columns[feature_loc:]]


def create_TestAgainstMyself_csv (input_dir, output_dir, minimun_pos_samples = 750):
    pos_files = list(input_dir.glob('*pos*.csv'))
    neg_files = list(input_dir.glob('*neg*.csv'))

    for f in pos_files:
        pos = pd.read_csv(f)
        if (pos.shape[0] < minimun_pos_samples):
            continue
        neg = pd.read_csv(neg_file(f, neg_files))
        pos_lite = select_feature(pos)
        neg_lite = select_feature(neg)
        pos_lite.insert(0, "Label", 1)
        neg_lite.insert(0, "Label", 0)
        pos_test, pos_train_all, pos_train_fixed = trainAll_trainfixed_test_split(df=pos_lite,
                                                                                  test_size=minimun_pos_samples)
        neg_test, neg_train_all, neg_train_fixed = trainAll_trainfixed_test_split(df=neg_lite,
                                                                                  test_size=minimun_pos_samples)

        test = pd.concat([pos_test, neg_test], ignore_index=True)
        train_all = pd.concat([pos_train_all, neg_train_all], ignore_index=True)
        train_fixed = pd.concat([pos_train_fixed, neg_train_fixed], ignore_index=True)

        print(f)
        out_f_suffix = f.stem.split("pos_")[1]
        for name, d in [("test", test), ("train_all", train_all), ("train_fixed", train_fixed)]:
            out_f = f"{name}_{out_f_suffix}.csv"
            d.to_csv(output_dir / out_f)


def get_feature_score (data, score_func, k="all"):
    X = data.iloc[:,1:]  #independent columns
    y = data.iloc[:,0]    #target column

    bestfeatures = SelectKBest(score_func=score_func,  k="all")
    fit = bestfeatures.fit(X,y)

    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    return featureScores

def plot_file (df_list, path_to_save, suff):

    ####################################3
    # Plot corr map
    ####################################3
    for i, d in enumerate(df_list):
        print(i)
        corrmat = d.corr()
        fig = sns.heatmap(corrmat, vmax=.8, square=True)
        fig.savefig(path_to_save / f"eta_{i}.{suff}")




def main():
    csv_dir = Path("Features/CSV")
    TestAgainstMyself_dir = Path("Features/CSV/TestAgainstMyself")

    ###########################################3
    # Create TestAgainstMyself files
    ###########################################3
    create_TestAgainstMyself_csv(csv_dir, TestAgainstMyself_dir)
    exit(3)
    ###########################################3
    # Plot corr matrix
    ###########################################3
    # df_list = []
    # for f_train in TestAgainstMyself_dir.glob('*train_fixed*'):
    # #for f_train in TestAgainstMyself_dir.glob('*train*'):
    #     df = pd.read_csv(f_train, index_col=0)
    #     if df.shape[0]<1000:
    #         continue
    #     df_list.append(df)
    # plot_file(df_list, TestAgainstMyself_dir, "pdf")
    # exit(3)

    ###########################################3
    # Calc the feature score for each train file
    ###########################################3
    summary = None
    # for f_train in TestAgainstMyself_dir.glob('*test*'):
    for f_train in TestAgainstMyself_dir.glob('*'):
    #for f_train in TestAgainstMyself_dir.glob('*train*'):
        df = pd.read_csv(f_train, index_col=0)
        if df.shape[0]<1000:
            continue
        if summary is None:
            summary = pd.DataFrame(index=df.columns)
        print(df.shape)

        feature_score = get_feature_score(df, score_func=mutual_info_classif)
        print(feature_score.shape)
        feature_score = feature_score[feature_score["Score"]!=0]
        print(feature_score.shape)

        cur_dataset = f_train.stem
        summary.insert(0, cur_dataset, 0)
        for feature in summary.index:
            if feature in list(feature_score["Specs"]):
                summary.loc[feature, cur_dataset]=feature_score[feature_score["Specs"]==feature]["Score"].item() #1
    summary["sum"] = summary.sum(axis=1)
    summary.sort_values(["sum"], axis=0, ascending=False, inplace=True)
    summary.to_csv(TestAgainstMyself_dir/ "summary.csv")
    summary = summary[summary["sum"]>0]
    print(summary.to_latex())
    best_features = pd.DataFrame(columns=summary.index)
    best_features.to_csv(TestAgainstMyself_dir/ "best_features.csv")








if __name__ == "__main__":
    main()
