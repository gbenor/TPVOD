from Pipeline import *
from pathlib import Path

from TPVOD_Utils import JsonLog, utils
import pandas as pd
import sys, ast


def read_paper_data(f, debug=False):
    if debug:
        return pd.read_csv(f, sep="\t", skiprows=30, nrows=50)
    return pd.read_csv(f, sep="\t", skiprows=30)


def df_prepare (in_df):
    in_df['GI_ID']= range(len(in_df))
    in_df.rename(columns={'miRNA_seq': 'miRNA sequence', 'mRNA_seq_extended': 'target sequence',
                          'chimeras_decompressed': 'number of reads'}, inplace=True)

    #  in_df.rename(columns={'miRNA ID': 'microRNA_name'}, inplace=True)
    return in_df




def main():
    try:
        debug=ast.literal_eval(sys.argv[1])
    except IndexError:
        debug=True

    if (debug):
        print ("***************************************\n"
               "\t\t\t DEBUG \n"
               "***************************************\n")

    interaction_file = str(Path("Papers/1-s2.0-S009286741300439X-mmc1.txt"))
    log_dir = "Datafiles_Prepare/Logs/"
    tmp_dir = utils.make_tmp_dir("Datafiles_Prepare/tmp_dir", parents=True)


    organisms = ["Human"]
    for organism in organisms:
        JsonLog.set_filename(
            utils.filename_date_append(Path(log_dir) / Path("Mapping_the_Human_miRNA_" + organism + ".json")))
        JsonLog.add_to_json('file name', interaction_file)
        JsonLog.add_to_json('paper',
                            "Mapping the Human miRNA Interactome by CLASH Reveals Frequent Noncanonical Binding")
        JsonLog.add_to_json('Organism', organism)
        JsonLog.add_to_json('paper_url', "https://www.sciencedirect.com/science/article/pii/S009286741300439X")
        p = Pipeline(paper_name="Mapping_the_Human_miRNA",
                     organism=organism,
                     in_df=df_prepare(read_paper_data(interaction_file, debug)),
                     tmp_dir=tmp_dir)

        p.run()

if __name__ == "__main__":
    main()
