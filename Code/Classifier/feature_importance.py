import pickle
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer

from utils import mean_std


DATASET_LIST = ['human_dataset1', 'human_dataset2', 'human_dataset3', 'mouse_dataset1',  'mouse_dataset2',
                'celegans_dataset1', 'celegans_dataset2', 'cattle_dataset1']
dataset_translation_list = {'human_dataset1': "h1", 'human_dataset2': "h2", 'human_dataset3': "h3",
                                    'mouse_dataset1': "m1", 'mouse_dataset2': "m2",
                                    'celegans_dataset1': "ce1", 'celegans_dataset2': "ce2",
                                    'cattle_dataset1': "ca1"}

DATASET_LIST.sort()

def generate_importance_files(id: int, prefix=""):
    results_dir = Path("Results") / f"{prefix}self{id}"
    method = 'xgbs_no_encoding'
    importance_types = ["weight", "gain", "cover", "total_gain", "total_cover"]
    with (Path("Results")/"490feature_list.txt").open("rb") as f:
        features = pickle.load(f)


    for model_file in results_dir.glob(f"*{method}*.model"):
        print (f"model file = {model_file}")
        feature_importance = pd.DataFrame (index=features)
        with model_file.open("rb") as f:
            clf = pickle.load(f)
        for it in importance_types:
            feature_importance[it] = pd.Series(clf.get_booster().get_score(importance_type=it))
        feature_importance.fillna(value=0, inplace=True)
        feature_importance.to_csv(results_dir / f"feature_importance_{model_file.stem}.csv")
        print("save feature importance file")


def feature_importance_table():
    writer = pd.ExcelWriter('Results/feature_importance.xlsx', engine='xlsxwriter') # supplementary_file
    for dataset in DATASET_LIST:
        print(dataset)
        mstd_df, m, _ = mean_std(dir=Path("Results"), match="self*", summary_shape=(490, 5),
                                 summary_file_name=f"feature_importance_{dataset}_xgbs_no_encoding.csv")
        m.to_csv (Path("Results")/ f"new_feature_importance_{dataset}.csv")
        mstd_df.sort_index().to_excel(writer, sheet_name=dataset_translation_list[dataset])
    writer.save()


def create_feature_df():
    dir = Path("Results")
    fi = pd.DataFrame()
    for f in dir.glob("*new*"):
        dataset = dataset_translation_list[f.stem.split("importance_")[1]]
        fi[dataset] = pd.read_csv(f, index_col=0)["gain"]
    return fi


def feature_importance_plot(fi):
    fi_sort = pd.DataFrame(columns=fi.columns)
    for c in fi_sort.columns:
        fi_sort[c] = fi[c].sort_values(ascending=False).values

    # Plot the full df
    fi_sort.plot()
    plt.xlabel('Features')
    plt.ylabel('Gain')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    plt.savefig(dir/ "feature_importance_full.pdf", format="pdf", bbox_inches='tight')

    # Plot the zoom df
    zoom_df = fi_sort.head(20)
    zoom_df.plot()
    plt.xlabel('Features')
    plt.ylabel('Gain')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)
    plt.xticks(range(0, len(zoom_df), 5))
    plt.savefig(dir / "feature_importance_zoom.pdf", format="pdf", bbox_inches='tight')


def max_transformer(X):
    scale = 100 / X.max(axis=0)
    return scale * X

def get_top_features (fi, n):
    top_features = set()
    for c in fi.columns:
        current_dataset_top_features = set(fi[c].sort_values(ascending=False).head(n).index)
        top_features = top_features.union(current_dataset_top_features)
    top_df = fi.loc[top_features,:]
    transformer = FunctionTransformer(max_transformer)
    top_df = transformer.transform(top_df)
    top_df["mean"] = top_df.mean(axis=1)
    top_df.sort_values(by="mean", ascending=False, inplace=True)
    top_df = top_df.round(0).astype(int)

    top_df.to_csv("Results/top_features.csv")


def main():
    # importance_types = ["weight", "gain", "cover", "total_gain", "total_cover"]
    # print (", ".join(importance_types))
    # exit(3)
    # for i in range(20):
    #     print (20*"*")
    #     print (i)
    #     generate_importance_files(i)
    feature_importance_table()
    fi = create_feature_df()
    # feature_importance_plot(fi)
    get_top_features(fi, 6)




if __name__ == '__main__':
    main()
