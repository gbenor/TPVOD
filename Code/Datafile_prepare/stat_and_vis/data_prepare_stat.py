from pathlib import Path
import json
import pandas as pd
import itertools
data_prepare_log_dir = Path("Datafiles_Prepare/Logs/")
data_prepare_csv_dir = Path("Datafiles_Prepare/CSV/")



def files_rename (filelist, suffix):
    datasets = {
        "Human_Mapping_the_Human_miRNA":        "human_dataset1",
        "Human_Unambiguous_Identification" :    "human_dataset2",
        "Human_Darnell_miRNA_target_chimeras":  "human_dataset3",
        "Celegans_Unambiguous_Identification_": "celegans_dataset1",
        "Celegans_Pairing_Beyond_Seed":         "celegans_dataset2",
        "Mouse_Unambiguous_Identification":     "mouse_dataset1",
        "Mouse_Darnell_miRNA_target" :          "mouse_dataset2",
        "Cow_Global_Mapping_Cattle":            "cattle_dataset1",

        "Mapping_the_Human_miRNA_Human":        "human_dataset1",
        "Unambiguous_Identification_Human":     "human_dataset2",
        "Darnell_miRNA_target_chimeras_Human":  "human_dataset3",
        "Unambiguous_Identification_Celegans":  "celegans_dataset1",
        "Pairing_Beyond_Seed_Celegans":         "celegans_dataset2",
        "Unambiguous_Identification_Mouse":     "mouse_dataset1",
        "Darnell_miRNA_target_chimeras_Mouse":  "mouse_dataset2",
        "Global_Mapping_Cattle_Cow":            "cattle_dataset1"
        }
    for f in filelist:
        print (f)
        for k, v in datasets.items():
            if f.match(f"*{k}*"):
                new_name = f.parent / f"{v}.{suffix}"
                print (f"rename {new_name}")
                f.rename(new_name)


def dataset_information ():
    min_num_of_pairs=11
    s = pd.DataFrame()

    for index, f in enumerate(data_prepare_log_dir.glob('*.json')):
        dataset = f.stem
        json_data = f.open().read()
        data = json.loads(json_data)


        pipe_in = data ["Pipeline input samples count"]
        valid_utr3 = data ["Pipeline valid blast results"]
        valid_mirna = data ["Pipeline valid miRNA_no_***"]

        s.loc ["Dataset", index] = dataset
        s.loc ["Samples", index] = pipe_in
        s.loc ["Valid_utr3", index] = min (valid_utr3, valid_mirna)

        dp_file = data_prepare_csv_dir / f"{dataset}.csv"
        df = pd.read_csv(dp_file)
        valid_duplex_df  = df[df["num_of_pairs"] >= min_num_of_pairs]
        valid_duplex = valid_duplex_df.shape[0]
        valid_seeds = sum(valid_duplex_df["valid_seed"])

        s.loc ["Valid duplex", index] = valid_duplex
        s.loc ["Valid seeds", index] = valid_seeds
        s.loc ["Final size", index] = valid_seeds*2

    s.sort_values(by='Dataset', axis=1, inplace=True)
    print (s)


    lat = s.to_latex(header=False, index=True)
    lat = lat.replace("\\_dataset", "")

    lat = lat.replace("\\textbackslash cite", "\\cite")
    lat = lat.replace("\\{", "{")
    lat = lat.replace("\\}", "}")
    lat = lat.replace("l}", "l|}")
    lat = lat.replace("{l", "{||l")
    lat = lat.replace("ll", "l|l|")
    lat = lat.replace("darnel|l|\\", "darnell")
    lat = lat.replace("global|", "global")

    caption = "\\caption{Summary of the data preparation pipeline}\n \
     \\label{tab:pipeline_summary}\n \
    \\end{tabular}"
    lat = lat.replace("\end{tabular}",caption)

    return (lat)

# ##########################################3333
# # Train-test split
# ##########################################3333
# def df_append(df, l):
#     return df.append(pd.DataFrame([l], columns=df.columns), ignore_index=True)
#
#
#
# datasets=[]
# for c in s.columns:
#     paper = s.loc["paper", c]
#     org = s.loc["Organism", c]
#     samples = s.loc["Valid seeds",c]*2
#     datasets.append((f"{org}_{paper}", samples))
#
# min_size = 1500
# dataset_paris = list(itertools.product(datasets, datasets))
# test_train_df=pd.DataFrame(columns=["Test","Test size" ,"Train", "Train fixed size", "Train full size"])
# for test, train in dataset_paris:
#     test_name, test_size = test
#     train_name, train_size = train
#     if test_size < min_size:
#         continue
#     if train_size < min_size:
#         continue
#     if test_name==train_name:
#         if train_size<min_size*2:
#             continue
#
#         test_train_df = df_append(test_train_df, [test_name, min_size, train_name, min_size, train_size-min_size])
#         # print (f"Train:\t{train_name}\tTrain_size:\t{train_size-min_size}\tTest:\t{test_name}")
#     else:
#         #print(f"Train:\t{train_name}\tTrain_size:\t{train_size}\tTest:\t{test_name}")
#         test_train_df = df_append(test_train_df, [test_name, min_size, train_name, min_size, train_size])
#         continue
# print (test_train_df)
#
# lat = test_train_df.to_latex()
# lat = lat.replace("\\textbackslash cite", "\\cite")
# lat = lat.replace("\\{", "{")
# lat = lat.replace("\\}", "}")
# lat = lat.replace("l}", "l|}")
# lat = lat.replace("{l", "{||l")
# lat = lat.replace("ll", "l|l|")
# lat = lat.replace("darnel|l|\\", "darnell")
# lat = lat.replace("global|", "global")
# lat = lat.replace("ful|l|", "full")
#
# print ("\n\n *************************************")
# print (lat)
#
#
#

def main():
    files_rename(data_prepare_csv_dir.glob('*duplex*.csv'), "csv")
    files_rename(data_prepare_log_dir.glob("*json"), "json")
    lat = dataset_information()
    print(lat)




if __name__ == "__main__":
    main()


