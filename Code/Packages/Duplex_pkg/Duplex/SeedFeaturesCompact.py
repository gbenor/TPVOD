from collections import Counter
from Duplex.InteractionRichPresentation import  *
from Duplex.SeedException import SeedException

import pandas as pd
import pprint


def count_not_space(s):
    return len(s) - s.count(' ')


class SeedFeaturesCompact(object):

    def __init__(self, seed, not_seed=False):


        SEED_SITE_SIZE = 8
        self.seed = seed
        self.seed.replace_T_U()
        mirna_nt_sum = count_not_space(seed.mir_inter) + count_not_space(seed.mir_bulge)
        assert (mirna_nt_sum==SEED_SITE_SIZE or not_seed), "seed size is wrong. it must be {} nt.\n{}".format(SEED_SITE_SIZE, seed)

        self.c = ['AU', 'UA', 'GC', 'CG']
        self.w = ['GU', 'UG']
        self.smt_dic = None

        self.mirna_last_nt = list(seed.mir_iterator())[-1][0]






    def extract_seed_features(self):
        self.seed_match_type()


    def get_features(self):
        f = [self.smt_dic]
        df = map(lambda x: pd.DataFrame([x]), f)
        r =  reduce (lambda x,y: pd.concat([x, y], axis=1, sort=False), df)
        return r

    def startingA (self):
        mirna0 = self.seed.mix_inter_bulge(self.seed.mir_inter[0], self.seed.mir_bulge[0])
        mrna0 = self.seed.mix_inter_bulge(self.seed.mrna_inter[0], self.seed.mrna_bulge[0])

        A_mirna = mirna0 == 'A' and mirna0 + mrna0 not in self.c
        A_mrna = mrna0 == 'A' and mirna0 + mrna0 not in self.c

        return 1 if (A_mirna or A_mrna) else 0

    def mismatch (self, i):
        GU = ['GU', 'UG']
        pair = self.seed.mir_bulge[i] + self.seed.mrna_bulge[i]
        if pair in GU:
            return 0
        if ' ' not in pair:
            return 1
        return 0
        #return 1 if self.seed.mir_bulge[i]!=' ' and  self.seed.mrna_bulge[i]!=' ' else 0

    def bulge (self, a,b):
        return sum([a[i]!=' ' and b[i]==' ' for i in range (len(a))])



    def countGU (self):
        cnt = 0

        for i in range(self.mirna_last_nt+1):
            s = self.seed.mir_bulge[i] + self.seed.mrna_bulge[i]
            if s in self.w:
                cnt+=1
        return cnt

    def count_interaction (self):
        return sum(1 for _ in self.seed.interaction_iterator())

    def startingIndex (self):
        for i in range (8):
            if self.seed.mir_inter[i]!=' ':
                return i
        raise SeedException("not valid seed. No interaction at all")

    def canonical_seed (self):
        c2_7 = self.smt_dic['Seed_match_compact_interactions_2_7']
        c3_8 = self.smt_dic['Seed_match_compact_interactions_3_8']
        bulge_mismatch_2_7 = max(count_not_space(self.s2_7.seed.mir_bulge), count_not_space(self.s2_7.seed.mrna_bulge))
        bulge_mismatch_3_8 = max(count_not_space(self.s3_8.seed.mir_bulge), count_not_space(self.s3_8.seed.mrna_bulge))
        if (bulge_mismatch_2_7>0) or (bulge_mismatch_3_8>0):
            return False
        return (c2_7==6) or (c3_8==6)

    def non_canonical_seed (self):
        c2_7 = self.smt_dic['Seed_match_compact_interactions_2_7'] + self.smt_dic['Seed_match_compact_GU_2_7']
        c3_8 = self.smt_dic['Seed_match_compact_interactions_3_8'] + self.smt_dic['Seed_match_compact_GU_3_8']
        bulge_mismatch_2_7 = max(count_not_space(self.s2_7.seed.mir_bulge), count_not_space(self.s2_7.seed.mrna_bulge)) - self.smt_dic['Seed_match_compact_GU_2_7']
        bulge_mismatch_3_8 = max(count_not_space(self.s3_8.seed.mir_bulge), count_not_space(self.s3_8.seed.mrna_bulge)) - self.smt_dic['Seed_match_compact_GU_3_8']
    
        r2_7 = (c2_7>=5) and (bulge_mismatch_2_7 <= 1)
        r3_8 = (c3_8>=5) and (bulge_mismatch_3_8 <= 1)

        return (r2_7 or r3_8) and (not self.canonical_seed())


    def valid_seed (self):
        #Valid Seed must have at least 6 combination of interactions and GUs
        assert self.smt_dic is not None, "The seed dict hasn't initiated yet."
        # return (self.smt_dic['Seed_match_compact_interactions'] + self.smt_dic['Seed_match_compact_GU']) >= 6
        return self.canonical_seed() or self.non_canonical_seed()


    def seed_match_type(self):
        smt_dic = {'Seed_match_compact_interactions_all': 0,
                   'Seed_match_compact_GU_all': 0,
                   'Seed_match_compact_interactions_2_7': 0,
                   'Seed_match_compact_GU_2_7': 0,
                   'Seed_match_compact_interactions_3_8': 0,
                   'Seed_match_compact_GU_3_8': 0,
                   'Seed_match_compact_A': 0,
                   'Seed_match_compact_start': 0,
                   'Seed_match_compact_mismatch_left': 0,
                   'Seed_match_compact_mismatch_right': 0,
                   'Seed_match_compact_mismatch_inner': 0,
                   'Seed_match_compact_bulge_target': 0,
                   'Seed_match_compact_bulge_mirna': 0,
                   # 'Seed_match_compact_canonical_seed': 0,
                   # 'Seed_match_compact_non_canonical_seed': 0,
                   }


        # smt_dic['Seed_match_compact_canonical_seed'] = self.canonical_seed()
        # smt_dic['Seed_match_compact_non_canonical_seed'] = self.non_canonical_seed()
        self.s2_7 = SeedFeaturesCompact(self.seed.extract_seed(2, 7), not_seed=True)
        self.s3_8 = SeedFeaturesCompact(self.seed.extract_seed(3, 8), not_seed=True)

        smt_dic['Seed_match_compact_interactions_all'] = self.count_interaction()
        smt_dic['Seed_match_compact_GU_all'] = self.countGU()

        smt_dic['Seed_match_compact_interactions_2_7'] = self.s2_7.count_interaction()
        smt_dic['Seed_match_compact_GU_2_7'] = self.s2_7.countGU()
        smt_dic['Seed_match_compact_interactions_3_8'] = self.s3_8.count_interaction()
        smt_dic['Seed_match_compact_GU_3_8'] = self.s3_8.countGU()

        # if smt_dic['Seed_match_compact_interactions'] + smt_dic['Seed_match_compact_GU'] < 6:
        #     raise SeedException("not valid seed. Seed must have at least 6 combination of interactions and GUs")

        smt_dic['Seed_match_compact_A'] = self.startingA()
        smt_dic['Seed_match_compact_start'] = self.startingIndex()

        smt_dic['Seed_match_compact_mismatch_left'] = self.mismatch(0)
        smt_dic['Seed_match_compact_mismatch_right'] = self.mismatch(self.mirna_last_nt)

        s_i = self.seed.mir_bulge.find(' ')
        s_e = self.seed.mir_bulge.rfind(' ')
        for i in range (s_i, s_e+1):
            smt_dic['Seed_match_compact_mismatch_inner'] += self.mismatch(i)

        smt_dic['Seed_match_compact_bulge_target'] = self.bulge(self.seed.mrna_bulge, self.seed.mir_bulge)
        smt_dic['Seed_match_compact_bulge_mirna'] = self.bulge(self.seed.mir_bulge, self.seed.mrna_bulge)

        # mir_bulges_count, mrna_bulges_count = self.seed.count_bulges()
        #
        #
        # # bulge = mir_bulges_count!=0 or mrna_bulges_count!=0
        # # if bulge:
        # #     if smt_dic['Seed_match_compact_interactions']<6:
        # #         raise SeedException("not valid seed. Seed with bulge must have at least 6 strong interactions")
        # smt_dic['Seed_match_compact_bulge_target'] = mrna_bulges_count
        # smt_dic['Seed_match_compact_bulge_mirna'] = mir_bulges_count
        #

        ###############################################################
        # Update the dict
        ###############################################################
        self.smt_dic = smt_dic



def test_seed (seed, seed_type):
    pp = pprint.PrettyPrinter(indent=4)
    print ("**************************************************")
    print ("Test: " + seed_type)
    print(seed)

    s = SeedFeaturesCompact(seed)
    s.extract_seed_features()
    print ("valid: {}".format(s.valid_seed()))


    pp.pprint (s.smt_dic)

    #assert seed_type in s.seed_type, "test error"


def main ():

    #########################################
    # Stringent
    #########################################

    s = InteractionRichPresentation("A       ",
                                    " GGGGGGG",
                                    " CCCCCCC",
                                    "C       ")
    test_seed(s, "Seed_match_8merA1")

    s = InteractionRichPresentation("A      C",
                                    " GGGGGG ",
                                    " CCCCCC ",
                                    "C      C")
    test_seed(s, "Seed_match_7merA1")

    s = InteractionRichPresentation("C       ",
                                    " GGGGGGG",
                                    " CCCCCCC",
                                    "C       ")
    test_seed(s, "Seed_match_7mer2")

    #########################################
    # Non-Stringent
    #########################################

    s = InteractionRichPresentation("C      C",
                                    " GGGGGG ",
                                    " CCCCCC ",
                                    "C      C")
    test_seed(s, "Seed_match_6mer2")

    s = InteractionRichPresentation("      CC",
                                    "GGGGGG  ",
                                    "CCCCCC  ",
                                    "      CC")
    test_seed(s, "Seed_match_6mer2")

    s = InteractionRichPresentation("CC      ",
                                    "  GGGGGG",
                                    "  CCCCCC",
                                    "CC      ")
    test_seed(s, "Seed_match_6mer2")

    s = InteractionRichPresentation("A    G  ",
                                    " GGGG GG",
                                    " CCCC CC",
                                    "C    U  ")
    test_seed(s, "Seed_match_6mer2GU1")

    s = InteractionRichPresentation("A    U  ",
                                    " GGGG GG",
                                    " CCCC CC",
                                    "C    G  ")
    test_seed(s, "Seed_match_6mer2GU1")

    s = InteractionRichPresentation("A  G U U",
                                    " GG G G ",
                                    " CC C C ",
                                    "C  U G G")
    test_seed(s, "Seed_match_6mer2GU3")


    s = InteractionRichPresentation("A    C  ",
                                    " GGGG GG",
                                    " CCCC CC",
                                    "C    C  ")
    test_seed(s, "Seed_match_6mer_LP")

    s = InteractionRichPresentation("A       ",
                                    " GGGG GG",
                                    " CCCC CG",
                                    "C    C  ")
    test_seed(s, "Seed_match_6mer_BM")

    s = InteractionRichPresentation("A C     U",
                                    " G GGGGG ",
                                    " C CCCCC ",
                                    "C       G")
    test_seed(s, "Seed_match_6mer_BT")

    s = InteractionRichPresentation("A CAG     U ",
                                    " G   GGGGG",
                                    " C   CCCCC",
                                    "C         G")
    test_seed(s, "Seed_match_6mer_BT")


if __name__ == "__main__":
    main()

