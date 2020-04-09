import pandas as pd
import numpy as np

from MatchingFeatures import *
from MrnaFeatures import *
from EnergyAccess import *
from pathlib import Path
import itertools
import multiprocessing
from NegativeSamples import NegativeSamples
from Duplex.ViennaRNADuplex import *
from pathlib import Path
from TPVOD_Utils import utils, JsonLog
from multiprocessing import Process
from Duplex.SeedException import SeedException
from Duplex.SeedFeaturesCompact import SeedFeaturesCompact
from Duplex.extract_site import extract_site
from config import CONFIG


def extract_features (row, tmp_dir):
    """

    Returns:
            valid_seed - Boolean - if this a valid seed
            num_of_pairs - Int - Num of pairs within the duplex
            feature_row - df - all the feature for this duplex

    """


    print (row.mRNA_name)
    # if not ("WBGene00011573|T07C12.9" in row.mRNA_name):
    #     return (False, np.nan, np.nan)




    ############################################################################
    # Generate the duplex
    ############################################################################
    dp = ViennaRNADuplex (row['miRNA sequence'], row['target sequence'], tmp_dir=tmp_dir)

    ############################################################################
    # Calc the site of interaction
    ############################################################################
    site, site_start_loc, site_end_loc = \
        extract_site (dp.IRP.site, row['target sequence'], row['mRNA_start'],
                      row['full_mrna'], row['mRNA_name'], dp.mrna_coor)
    print ("site start: " + str(site_start_loc))
    dp.IRP.set_site(site[::-1])
    print (dp)


    #####################
    # Seed XX Features
    #####################
    c_seed = dp.IRP.extract_seed()


    try:
        seed_feature_compact = SeedFeaturesCompact(c_seed)
        seed_feature_compact.extract_seed_features()
        valid_seed_compact = seed_feature_compact.valid_seed()
    except SeedException:
        valid_seed_compact = False
        #No seed. This is invalid duplex
        return (False, dp.num_of_pairs, np.nan)



    valid_seed = valid_seed_compact
    #####################
    # Matching & Pairing Features
    #####################
    matching_features = MatchingFeatures(dp.IRP)
    matching_features.extract_matching_features()

    #####################
    # mRNA Features
    #####################
    mrna_features = MrnaFeatures(row['miRNA sequence'], site, site_start_loc, site_end_loc,
                                 row['full_mrna'], flank_number=70)
    mrna_features.extract_mrna_features()

    #####################
    # Free energy & Accessibility
    #####################
    enrgy_access = EnergyAccess (dp, site, site_start_loc, site_end_loc, row['full_mrna'], tmp_dir=tmp_dir)

    #####################
    # Data frame construction
    #####################
    features = [seed_feature_compact.get_features(),
                #seed_feature.get_features(),
                matching_features.get_features(),
                mrna_features.get_features(),
                enrgy_access.get_features()
           ]
    features_df = reduce (lambda x,y: pd.concat([x, y], axis=1, sort=False), features)

    information = pd.DataFrame(row).transpose()
    information['site_start'] = site_start_loc
    try:
        information = reduce (lambda x,y: pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1, sort=False), [information, dp.get_features()])
    except AttributeError:
        # 'MirandaDuplex' object has no attribute 'get_features'
        pass


    new_line_df = reduce (lambda x,y: pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1, sort=False), [information, features_df])
    # cons = conservation(mrna, mr_site_loc)

    # mmp = pd.DataFrame([self.mmp_dic])
    # mpc = pd.DataFrame([self.mpc_dic])

    #valid_seed, num_of_pairsm, feature_row
    return (valid_seed, dp.num_of_pairs, new_line_df)




def worker (fin, fout, tmp_dir):
    min_num_of_pairs = CONFIG["minimum_pairs_for_interaction"]
    print ("Starting worker #{}".format(multiprocessing.current_process()))
    ########################
    # Read the input file
    ########################
    in_df = pd.read_csv(fin)
    cond1 = in_df["canonic_seed"]
    cond2 = (in_df["non_canonic_seed"])
    #& (in_df["num_of_pairs"] >= min_num_of_pairs)
    print (fin)
    in_df = in_df[(cond1) | (cond2)]
    print(in_df.shape)

    ########################
    # Create pos & neg dataframes
    ########################
    out_df = pd.DataFrame()

    i=0
    for index, row in in_df.iterrows():
        print(f"$$$$$$$$$$$$$$$ {i} $$$$$$$$$$$$$$$$$$4")
        i+=1
        # if i > 10:
        #     break


        valid_seed, num_of_pairs, feature_row = extract_features(row, tmp_dir)
        assert valid_seed, f"All seeds must be valid\n index={index}"
        # assert num_of_pairs >= min_num_of_pairs, f"All interaction must have atleast 11 pairs\n index={index}"
        out_df = pd.concat([out_df, feature_row], sort=False)

    ########################
    # Save df to CSV
    ########################
    out_df.reset_index(drop=True, inplace=True)
    utils.drop_unnamed_col(out_df)
    out_df.to_csv(fout)




def main():
    input_dir = Path("Datafiles_Prepare/CSV")
    output_dir = Path("Features/CSV")
    log_dir = Path("Features/Logs")
    tmp_base = "Features/tmp_dir"

    files = list(input_dir.glob("*_duplex_*.csv"))


    process_list = []
    for fin in files:
        tmp_dir = utils.make_tmp_dir(tmp_base, parents=True)
        fout = output_dir / f"{fin.stem}_feature.csv"
        if fout.exists():
            continue

        flog = log_dir / f"{fin.stem}_feature.json"

        p = Process(target=worker, args=(fin, fout, tmp_dir))
        p.start()
        process_list.append(p)
        print(f"start process {p.name} {fin}")
    for p in process_list:
        p.join()





if __name__ == "__main__":
    main()

