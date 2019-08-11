import pandas as pd
import numpy as np
from Duplex.ViennaRNADuplex import *
from Bio import SeqIO
import random
from Duplex.SeedFeatures import *
from Duplex.SeedFeaturesCompact import *
import MirBaseUtils.mirBaseUtils as MBU


from Duplex.InteractionRichPresentation import *

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord






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
            valid_seed_compact = seed_feature_compact.valid_seed()
        except SeedException:
            valid_seed_compact = False
            # No seed. This is invalid duplex
            return (False, dp.num_of_pairs)

        return valid_seed_compact , dp.num_of_pairs

    def generate_negative_seq (self, orig_mirna, full_mrna,
                               num_of_tries=10000):

        for i in range(num_of_tries):
            mock_mirna = self.generate_mirna_mock(orig_mirna)
            valid_seed, num_of_pairs = self.valid_negative_seq (mock_mirna, full_mrna)
            if valid_seed:
                if num_of_pairs >= self.min_num_of_pairs:
                    return True, mock_mirna, full_mrna

        return (False, np.nan, np.nan)
