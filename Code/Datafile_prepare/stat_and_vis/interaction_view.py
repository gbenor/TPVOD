import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_file(fin, min_num_of_pairs, suff):
    df = pd.read_csv(fin)
    flags_col = ['num_of_pairs', 'valid_seed']
    interaction_cols = ['i_0', 'i_1', 'i_2', 'i_3', 'i_4', 'i_5', 'i_6', 'i_7',
                        'i_8', 'i_9', 'i_10', 'i_11', 'i_12', 'i_13', 'i_14', 'i_15', 'i_16',
                        'i_17', 'i_18', 'i_19', 'i_20', 'i_21']

    col_to_select = list(itertools.chain(*[flags_col, interaction_cols]))
    newdf = df[col_to_select]

    ####################################3
    # Create figure
    ####################################3
    f = plt.figure(figsize=(6, 6))

    ####################################3
    # Sorted View
    ####################################3
    s = newdf.sort_values(by=['num_of_pairs'], ascending=False)
    img_s = s[interaction_cols].fillna(0).values
    ax = plt.subplot(1, 5, 1)
    ax.set_title('A')

    plt.imshow(img_s, aspect="auto")
    plt.rcParams["axes.grid"] = False

    ####################################3
    # View duplexes with at least min_num_of_pairs pairs
    ####################################3
    t = s[s["num_of_pairs"] >= min_num_of_pairs]
    z = np.zeros((s.shape[0] - t.shape[0], 22))
    r = t[interaction_cols].fillna(0).values
    img_tz = np.concatenate((r, z))
    ax = plt.subplot(1, 5, 2)
    ax.set_title('B')

    plt.imshow(img_tz, aspect="auto")
    plt.rcParams["axes.grid"] = False

    ####################################3
    # View in cluster
    ####################################3
    kmeans = KMeans(n_clusters=5, random_state=0)
    y = t.copy()
    y.reset_index(inplace=True)
    y["label"] = pd.Series(kmeans.fit_predict(t[interaction_cols].fillna(0)))
    ys = y.sort_values(by=['label'], ascending=True)
    ys.reset_index(inplace=True)
    gl = np.zeros((ys.shape[0], 22))
    for index, row in ys.iterrows():
        gl[index, :] = (row['label'] + 1) * (row[interaction_cols].values)
    yss = ys[interaction_cols]
    img_y = np.concatenate((yss.values, z))
    ax = plt.subplot(1, 5, 3)
    ax.set_title('C')

    plt.imshow(img_y, aspect="auto")
    plt.rcParams["axes.grid"] = False

    ax = plt.subplot(1, 5, 4)
    ax.set_title('D')

    img_gl = np.concatenate((gl, z))
    # plt.imshow(np.where(img_y>=1, 1, 0), aspect="auto")
    plt.imshow(img_gl, aspect="auto")

    ####################################3
    # Remove invalid Seeds
    ####################################3
    valid_seeds = ys[ys['valid_seed']]
    valid_seeds_sort = valid_seeds.sort_values(by=['label'], ascending=True)
    valid_seeds_sort2 = valid_seeds_sort[col_to_select]
    valid_seeds_sort2['label'] = valid_seeds_sort['label']
    valid_seeds_sort = valid_seeds_sort2.copy()
    valid_seeds_sort.reset_index(inplace=True)
    gl = np.zeros((valid_seeds_sort.shape[0], 22))
    for index, row in valid_seeds_sort.iterrows():
        gl[index, :] = (row['label'] + 1) * (row[interaction_cols].values)

    newz = np.zeros((s.shape[0] - valid_seeds_sort.shape[0], 22))
    img_y = np.concatenate((gl, newz))

    ax = plt.subplot(1, 5, 5)
    ax.set_title('E')

    plt.imshow(img_y, aspect="auto")
    plt.rcParams["axes.grid"] = False

    path_to_save = Path(fin).parent / ".."/ "Figures" /Path(fin.stem).with_suffix(f".{suff}")



    plt.savefig(path_to_save, format=suff, bbox_inches='tight')

def main():
    print(plt.style.available)
    plt.style.use('seaborn')
    suff = "pdf"


    f =  Path("Datafiles_Prepare/CSV")
    duplex_files = list (f.glob("*dataset*"))
    for a in duplex_files:
        print (a)
        plot_file (fin=a, min_num_of_pairs=11, suff=suff)

    print ("*****************************************************")
    print ("*****************************************************")

    f = Path("Datafiles_Prepare/Figures")
    duplex_files = list(f.glob("*.pdf"))
    for a in duplex_files:
        print ("\\begin{figure}[h!]")
        print(f"\t\includegraphics[width = 0.4\\textwidth]{{data_prepare/{a.stem}.{suff}}}")
        dataset = a.stem
        print (f"\t\\caption {{{dataset} miRNA interaction view.\\\\"
               f"A: All interactions sorted according the duplex strength\\\\"
               f"B: Interaction with at least 11 pairs\\\\"
               f"C: Arranging the interaction in 5 clusters\\\\"
               f"D: Different color for each cluster\\\\"
               f"E: Showing only valid seeds}}")
        print (f"\t\\label{{fig:{a.stem}}}")
        print ("\\end {figure}")
        print ()


if __name__ == "__main__":
    main()







