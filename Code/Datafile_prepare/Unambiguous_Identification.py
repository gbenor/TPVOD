from Pipeline import *
from pathlib import Path
from TPVOD_Utils import JsonLog, utils
import pandas as pd
import sys, ast


def read_paper_data(f, organism, debug=False):
    sheets = {"elegans" : "Sheet2",
               "human" : "Sheet3",
               "viral" : "Sheet4",
               "mouse" : "Sheet5"}

    organism = organism.lower()
    if organism == "celegans" :
        organism = "elegans"

    current_sheet = sheets[organism]

    # Read the interaction file
    inter_df = pd.read_excel(f, sheet_name=current_sheet, header=None)
    assert organism in inter_df.iloc[0, 0].lower(), "Read the wrong sheet. no {} in the first cell".format(organism)
    if debug:
        return pd.read_excel(f, sheet_name=current_sheet, skiprows=3)[:50]
    return pd.read_excel(f, sheet_name=current_sheet, skiprows=3)


def df_prepare (in_df):
    in_df['GI_ID']= range(len(in_df))
    in_df.rename(columns={'miRNA ID': 'microRNA_name'}, inplace=True)
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

    interaction_file = str(Path("Papers/1-s2.0-S1097276514003566-mmc3.xls"))
    log_dir = "Datafiles_Prepare/Logs/"
    tmp_dir = utils.make_tmp_dir("Datafiles_Prepare/tmp_dir", parents=True)

    organisms = ["Celegans", "Human",  "Mouse"]
    for organism in organisms:
        JsonLog.set_filename(utils.filename_date_append(Path(log_dir) / Path("Unambiguous_Identification_" + organism + ".json")))
        JsonLog.add_to_json('file name', interaction_file)
        JsonLog.add_to_json('paper',
                            "Unambiguous Identification of miRNA:Target Site Interactions by Different Types of Ligation Reactions")
        JsonLog.add_to_json('Organism', organism)
        JsonLog.add_to_json('paper_url', "https://www.sciencedirect.com/science/article/pii/S1097276514003566#app3")
        p = Pipeline(paper_name="Unambiguous_Identification",
                    organism=organism,
                    in_df=df_prepare(read_paper_data(interaction_file, organism, debug)),
                    tmp_dir=tmp_dir)

        p.run()


if __name__ == "__main__":
    main()


