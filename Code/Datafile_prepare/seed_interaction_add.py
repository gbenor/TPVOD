import pandas as pd
duplex_method = "vienna"
from Duplex.ViennaRNADuplex import *
from pathlib import Path
from TPVOD_Utils import utils
from multiprocessing import Process
from Duplex.SeedException import SeedException
from Duplex.SeedFeaturesCompact import SeedFeaturesCompact

def seed_interaction_add (fin, fout, tmp_dir):
    i=0
    relevant_col = ["Source", "Organism",  "GI_ID", "microRNA_name", "miRNA sequence", "target sequence",
                    "number of reads", "mRNA_name",	"mRNA_start", "mRNA_end", "full_mrna"]
    df = pd.read_csv(fin, usecols=relevant_col)
    for index, row in df.iterrows():
        # i += 1
        # if i> 50:
        #     break
        ############################################################################
        # Generate the duplex
        ############################################################################
        if duplex_method == "vienna":
            dp = ViennaRNADuplex(row['miRNA sequence'], row['target sequence'], tmp_dir=tmp_dir)
        # if duplex_method == "vienna_with_constraint":
        #     dp = ViennaRNADuplex(row['miRNA sequence'], row['target sequence'],
        #                          constraint=".||||")  # nt2-5 must be paired
        # if duplex_method == "miranda":
        #     try:
        #         dp = MirandaDuplex(row['miRNA sequence'], row['target sequence'], "Data/Human/Parsed")
        #     except NoMirandaHits:
        #         return (False, 0, np.nan)

        #####################
        # Seed Extraction
        #####################
        c_seed = dp.IRP.extract_seed()

        try:
            seed_feature_compact = SeedFeaturesCompact(c_seed)
            seed_feature_compact.extract_seed_features()
            valid_seed_compact = seed_feature_compact.valid_seed()
            canonic_seed = seed_feature_compact.canonical_seed()
            non_canonic_seed = seed_feature_compact.non_canonical_seed()
        except SeedException:
            valid_seed_compact = False # No seed. This is invalid duplex
            canonic_seed = False
            non_canonic_seed = False
        ############################################################################
        # Getting the information
        ############################################################################


        df.loc[index, "num_of_pairs"] = dp.num_of_pairs
        df.loc[index, "valid_seed"] = valid_seed_compact
        df.loc[index, "canonic_seed"] = canonic_seed
        df.loc[index, "non_canonic_seed"] = non_canonic_seed

        for k, pair in enumerate(dp.IRP.mir_pairing_iterator()):
            int_flag = 1 if " " not in pair else 0
            df.loc[index, "i_{}".format(k)] = int_flag

    ############################################################################
    # save the file
    ############################################################################

    df.to_csv(fout)



def main ():
    p = Path("Datafiles_Prepare/CSV")
    log_dir = "Datafiles_Prepare/Logs/"

    files = list(p.glob('**/*.csv'))

    run_list = [p for p in files if not p.match("*duplex*")]
    process_list = []
    for fin in run_list:
        fout = utils.filename_suffix_append(fin, "_duplex")
        tmp_dir = utils.make_tmp_dir("Datafiles_Prepare/tmp_dir", parents=True)

        p = Process(target=seed_interaction_add, args=(fin, fout, tmp_dir))
        p.start()
        process_list.append(p)
        print (f"start process {p.name} {fin}")

    for p in process_list:
        p.join()

if __name__ == "__main__":
    main()


