from pathlib import Path

import pandas as pd
import numpy as np
from TPVOD_Utils import utils

from Duplex.ViennaRNADuplex import *
from Bio import SeqIO
import random
from Duplex.SeedFeatures import *
from Duplex.SeedFeaturesCompact import *
import MirBaseUtils.mirBaseUtils as MBU
from multiprocessing import Process


from Duplex.InteractionRichPresentation import *

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from TPVOD_Utils.utils import filename_suffix_append
from config import CONFIG


class NegativeSamples(object):

    def __init__(self, organism, tmp_dir, min_num_of_pairs):
        self.tmp_dir = tmp_dir
        self.min_num_of_pairs = min_num_of_pairs
        mirna_prefix_table = {"human" : "hsa",
                               "mouse" : "mmu",
                               "elegans" : "cel",
                               "celegans": "cel",
                               "cattle": "bta"}

        miRBase_dic = MBU.read_mirbase_file(mirna_prefix_table[organism.lower()])
        self.miRBase_seq_list = miRBase_dic["mi_seq"].tolist()
        #self.miRBase_seq_list = [v[1] for k,v in miRBase_dic.items()]


    # # 4. a function used to generate mock mirna which their seed never equal to any mirna in the miRBase.
    def generate_mirna_mock(self, mirna, th=5):
        def seed_equal(a, b):
            return sum(a[i] == b[i] for i in range(len(a)))

        def equal_to_mirbase (mir, th=5):
            for seq in self.miRBase_seq_list:
                seq = list(seq)
                e27 = seed_equal(mir[1:7], seq[1:7])
                if e27 > th:
                    return True
                e38 = seed_equal(mir[2:8], seq[2:8])
                if e38 > th:
                    return True
            return False


        mirna_list_o = list (mirna.replace('T', 'U').upper())
        mirna_list_r = list (mirna.replace('T', 'U').upper())
        equal_to_itself = True

        num_shuffle = 0
        while equal_to_itself or eq_mirbase:
            random.shuffle(mirna_list_r)
            num_shuffle += 1
            if num_shuffle % 10000 == 0:
                print (num_shuffle)
            if num_shuffle > 100000:
                break

            # check if it equals to itself
            e27 = seed_equal(mirna_list_r[1:7], mirna_list_o[1:7])
            e38 = seed_equal(mirna_list_r[2:8], mirna_list_o[2:8])
            equal_to_itself = e27 > th or e38 > th
            # check against mirbase
            eq_mirbase = equal_to_mirbase(mirna_list_r)


        mirna_m = ''.join(mirna_list_r)
        return mirna_m.replace('U','T')


    def valid_negative_seq(self, mir, mrna):
        dp = ViennaRNADuplex(mir, mrna, tmp_dir=self.tmp_dir)
        c_seed = dp.IRP.extract_seed()

        try:
            seed_feature_compact = SeedFeaturesCompact(c_seed)
            seed_feature_compact.extract_seed_features()
            canonic_seed = seed_feature_compact.canonical_seed()
            non_canonic_seed = seed_feature_compact.non_canonical_seed()
        except SeedException:
            canonic_seed = False
            non_canonic_seed = False
        return canonic_seed, non_canonic_seed, dp.num_of_pairs

    def generate_negative_seq (self, orig_mirna, full_mrna,
                               num_of_tries=10000):

        for i in range(num_of_tries):
            mock_mirna = self.generate_mirna_mock(orig_mirna)
            canonic_seed, non_canonic_seed, num_of_pairs = self.valid_negative_seq (mock_mirna, full_mrna)
            cond1 = canonic_seed
            cond2 = non_canonic_seed
            #and num_of_pairs >= self.min_num_of_pairs
            if cond1 or cond2:
                properties = {
                    "mock_mirna" : mock_mirna,
                    "full_mrna" : full_mrna,
                    "canonic_seed" : canonic_seed,
                    "non_canonic_seed" : non_canonic_seed,
                    "num_of_pairs" : num_of_pairs
                }
                return True, properties
        return False, {}

def worker (organism, fin, tmp_dir):
    fout = filename_suffix_append(fin,"_negative")
    min_num_of_pairs = CONFIG["minimum_pairs_for_interaction"]
    ns = NegativeSamples(organism, tmp_dir=tmp_dir, min_num_of_pairs=min_num_of_pairs)

    in_df = pd.read_csv(fin)
    cond1 = in_df["canonic_seed"]
    # cond2 = (in_df["non_canonic_seed"]) & (in_df["num_of_pairs"] >= min_num_of_pairs)
    cond2 = (in_df["non_canonic_seed"])

    in_df = in_df[(cond1) | (cond2)]

    neg_df = pd.DataFrame()

    i=0
    for index, row in in_df.iterrows():
        print(f"$$$$$$$$$$$$$$$ {i} $$$$$$$$$$$$$$$$$$4")
        i+=1
        valid, properties = ns.generate_negative_seq(row['miRNA sequence'],row['full_mrna'])
        if not valid:
            continue

        new_row = pd.Series()
        new_row['Source'] = row['Source']
        new_row['Organism'] = row['Organism']
        new_row['microRNA_name'] = "mock " + row.microRNA_name
        new_row['miRNA sequence'] = properties["mock_mirna"]
        new_row['target sequence'] = row['full_mrna']
        new_row['number of reads'] = row['number of reads']
        new_row['mRNA_name'] = row.mRNA_name
        new_row['mRNA_start'] = 0
        new_row['mRNA_end'] = len(row['full_mrna'])
        new_row['full_mrna'] = row['full_mrna']
        new_row["canonic_seed"] = properties["canonic_seed"]
        new_row["non_canonic_seed"] = properties["non_canonic_seed"]
        new_row["num_of_pairs"] = properties["num_of_pairs"]

        neg_df = neg_df.append(new_row, ignore_index=True)
        # if i> 10:
        #     break
        #



    ########################
    # Save df to CSV
    ########################

    print (neg_df.head())

    neg_df.reset_index(drop=True, inplace=True)
    utils.drop_unnamed_col(neg_df)
    neg_df.to_csv(fout)


def main():

    input_dir = Path("Datafiles_Prepare/CSV")

    tmp_base = "Features/tmp_dir"

    files = list(input_dir.glob("*duplex*.csv"))
    files = [f for f in files if not f.match("*negative*")]
    print (files)

    process_list = []
    for fin in files:
        tmp_dir = utils.make_tmp_dir(tmp_base, parents=True)

        organism = fin.stem.split("_")[0]
        print (organism)

        p = Process(target=worker, args=(organism, fin, tmp_dir))
        p.start()
        process_list.append(p)
        print(f"start process {p.name} {fin}")

    for p in process_list:
        p.join()




if __name__ == "__main__":
    main()




