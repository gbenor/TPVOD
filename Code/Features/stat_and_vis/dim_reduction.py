from collections import Counter
from functools import partial

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
interactions_dir = Path("Features/CSV")
figure_dir = Path("dim_reduction")

from multiprocessing import Pool
dataset_translation_list = {'human_dataset1' : "h1",'human_dataset2' : "h2",'human_dataset3' : "h3",
                            'mouse_dataset1' : "m1",'mouse_dataset2' : "m2",
                            'celegans_dataset1' : "ce1", 'celegans_dataset2' : "ce2",
                            'cattle_dataset1' : "ca1"}


def get_dataset_name (f: Path) -> str:
    return f.stem.split("_duplex")[0]


def interaction_loader(f: Path) -> pd.DataFrame:
    df = pd.read_csv(f)
    int_cols = [c for c in df.columns if str(c).startswith("i_")]
    int_cols = [c for c in int_cols if int(c.split("_")[1]) < 22]
    df = df[int_cols]
    return df.fillna(value=0)


def feature_loader(f: Path) -> pd.DataFrame:
    df = pd.read_csv(f)
    col_list = list(df.columns)
    all_features = col_list[col_list.index("Seed_match_compact_A"):]
    desired_features = [f for f in all_features if not str(f).startswith("HotPairing")]
    return df[desired_features]



LOADERS_DICT = {
    "interactions": interaction_loader,
    "features": feature_loader
}


def load_datasets (type: str):
    loader = LOADERS_DICT[type]
    df_list = []
    for f in interactions_dir.glob("*positive*csv"):
        name = get_dataset_name(f)
        print("Dataset: " + name)
        df = loader(f)
        df.insert(0, "dataset", name)
        df_list.append(df)
    result = pd.concat(df_list)
    return result


def random_resampler(df: pd.DataFrame):
    X = df.drop('dataset', axis=1)
    y = df['dataset']

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res


def plot2d(X: pd.DataFrame, y: pd.Series, figure_filename: Path):
    d_col = {"cattle": 0,
             "celegans": 1,
             "human": 2,
             "mouse": 3}

    fig = plt.figure(figsize=(16, 10))
    x_min_val =  X.iloc[:,0].min()
    x_max_val =  X.iloc[:,0].max()
    y_min_val = X.iloc[:,1].min()
    y_max_val = X.iloc[:,1].max()
    x_axis_range=[np.floor(x_min_val), np.ceil(x_max_val)]
    y_axis_range=[np.floor(y_min_val), np.ceil(y_max_val)]

    for title in y.unique():
        print (title)
        x2d = X[y == title]
        x2d.columns = ["2d-one", "2d-two"]
        org = title.split("_data")[0]
        row = int(title.split("dataset")[1]) - 1
        col = d_col[org]
        index = row * 4 + col + 1

        ax = fig.add_subplot(3, 4, index, title=dataset_translation_list[title])
        plt.xlim(x_axis_range)
        plt.ylim(y_axis_range)
        sns.scatterplot(
            x="2d-one", y="2d-two",
            palette=sns.color_palette("hls", 10),
            data=x2d,
            legend=False,
            alpha=0.3
        )
        # Set x-axis label
        plt.xlabel('')
        # Set y-axis label
        plt.ylabel('')

    plt.savefig(figure_filename, format="pdf", bbox_inches='tight')
    print (f"save to file: {figure_filename}")

def dim_reduction(X, y, final_len, reductor):
    result_x = pd.DataFrame(reductor.fit_transform(X)).head(final_len)
    result_y = y.head(final_len)
    print (reductor.explained_variance_ratio_)
    return result_x.reset_index(drop=True), result_y.reset_index(drop=True)

def worker(t, output_shape):
    (x1, y1), r_func, filename = t
    if filename.find("tsne")!=-1:
        return
    print (f"file {filename}")
    x2d, y2d = dim_reduction(x1, y1, output_shape, r_func)
    plot2d(x2d, y2d, figure_dir/filename)


def main():
    # print ("interactions_main")
    #
    # datasets = load_datasets("interactions")
    # y = datasets["dataset"]
    # x = datasets.drop('dataset', axis=1)
    # x_res, y_res = random_resampler(datasets)
    #
    # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=30000)
    # pca = PCA(n_components=2)
    #
    # tasks = [((x,y), pca, "interactions_pca_without_resample.pdf"),
    #          ((x, y), tsne, "interactions_tsne_without_resample.pdf"),
    #          ((x_res, y_res), pca, "interactions_pca_with_resample.pdf"),
    #          ((x_res, y_res), tsne, "interactions_tsne_with_resample.pdf")]
    #
    #
    # for t in tasks:
    #     my_worker(t)

    print ("features_main")
    datasets = load_datasets("features")
    y = datasets["dataset"]
    x = datasets.drop('dataset', axis=1)
    x_res, y_res = random_resampler(datasets)
    my_worker = partial(worker, output_shape=len(x))

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=30000)
    pca = PCA(n_components=2)

    tasks = [
        # ((x, y), pca, "features_pca_without_resample.pdf"),
            ((x_res, y_res), pca, "features_pca_with_resample.pdf"),
        #      ((x_res, y_res), tsne, "features_tsne_with_resample.pdf"),
        #        ((x, y), tsne, "features_tsne_without_resample.pdf")
    ]

    for t in tasks:
        my_worker(t)


if __name__ == '__main__':
    main()


