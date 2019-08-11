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



CONFIG = {
    'minimum_pairs_for_interaction': 11,
    'duplex_method' :  ["vienna"],
    'max_process' : 1
 #   'duplex_method': ["vienna", "miranda", "vienna_with_constraint"]

    #'duplex_method' :  ["miranda"]
}



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




def worker (organism, fin, pos_fout, neg_fout, flog, tmp_dir):
    min_num_of_pairs = CONFIG["minimum_pairs_for_interaction"]
    print ("Starting worker #{}".format(multiprocessing.current_process()))

    ########################
    # Read the input file, initiate the helper classes
    ########################
    ns = NegativeSamples(organism, tmp_dir=tmp_dir, min_num_of_pairs=min_num_of_pairs)

    in_df = pd.read_csv(fin)
    in_df = in_df[in_df['valid_seed']]
    in_df = in_df[in_df["num_of_pairs"] >= min_num_of_pairs]

    ########################
    # Create pos & neg dataframes
    ########################
    pos_df = pd.DataFrame()
    neg_df = pd.DataFrame()

    i=0
    for index, row in in_df.iterrows():
        print(f"$$$$$$$$$$$$$$$ {i} $$$$$$$$$$$$$$$$$$4")
        i+=1

        ########################
        # Handle positive sample
        ########################
        valid_seed, num_of_pairs, feature_row = extract_features(row, tmp_dir)
        assert valid_seed, f"All seeds must be valid\n index={index}"
        assert num_of_pairs >= min_num_of_pairs, f"All interaction must have atleast 11 pairs\n index={index}"
        pos_df = pd.concat([pos_df, feature_row], sort=False)

        ########################
        # Handle negative sample
        ########################

        valid, mock_mirna, full_mrna = \
            ns.generate_negative_seq(row['miRNA sequence'],row['full_mrna'])

        if not valid:
            continue


        row_for_extract_features = pd.Series()
        row_for_extract_features['Source'] = row['Source']
        row_for_extract_features['Organism'] = row['Organism']
        row_for_extract_features['microRNA_name'] = "mock " + row.microRNA_name
        row_for_extract_features['miRNA sequence'] = mock_mirna
        row_for_extract_features['target sequence'] = row['full_mrna']
        row_for_extract_features['number of reads'] = row['number of reads']
        row_for_extract_features['mRNA_name'] = row.mRNA_name
        row_for_extract_features['mRNA_start'] = 0
        row_for_extract_features['mRNA_end'] = len(row['full_mrna'])
        row_for_extract_features['full_mrna'] = row['full_mrna']

        valid_seed, nop, feature_row = extract_features(row_for_extract_features, tmp_dir=tmp_dir)
        assert valid_seed, "it must be valid seed since we demand it in generate_negative_seq"
        assert nop >= min_num_of_pairs, "it must be valid duplex since we have checked it in generate_negative_seq"

        neg_df = pd.concat([neg_df, feature_row], sort=False)


    ########################
    # Save df to CSV
    ########################
    for df_tuple in [(pos_df, pos_fout), (neg_df, neg_fout)]:
        df = df_tuple[0]
        fout =  df_tuple[1]
        df.reset_index(drop=True, inplace=True)
        utils.drop_unnamed_col(df)
        df.to_csv(fout)




def main():
    input_dir = Path("Datafiles_Prepare/CSV")
    output_dir = Path("Features/CSV")
    log_dir = Path("Features/Logs")
    tmp_base = "Features/tmp_dir"

    files = list(input_dir.glob("*dataset*.csv"))


    process_list = []
    for fin in files:
        tmp_dir = utils.make_tmp_dir(tmp_base, parents=True)
        pos_fout = output_dir / f"pos_{fin.stem}.csv"
        neg_fout = output_dir / f"neg_{fin.stem}.csv"

        flog = log_dir / f"pos_{fin.stem}.json"
        organism = fin.stem.split("_")[0]

        p = Process(target=worker, args=(organism, fin, pos_fout, neg_fout, flog, tmp_dir))
        p.start()
        process_list.append(p)
        print(f"start process {p.name} {fin}")
    for p in process_list:
        p.join()





if __name__ == "__main__":
    main()

