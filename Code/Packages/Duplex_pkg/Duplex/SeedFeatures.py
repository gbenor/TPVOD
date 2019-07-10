from collections import Counter
from InteractionRichPresentation import *
import pandas as pd

class SeedFeatures(object):

    def __init__(self, seed):
        self.seed = seed
        self.seed.replace_T_U()


    def extract_seed_features(self):
        self.seed_match_type()


    def get_features(self):
        f = [self.smt_dic]
        df = map(lambda x: pd.DataFrame([x]), f)
        r =  reduce (lambda x,y: pd.concat([x, y], axis=1, sort=False), df)
        return r


    def seed_complementary(self, seq1, seq2):
        count_c = 0
        count_w = 0
        count_mismatch = 0
        c = ['AU', 'UA', 'GC', 'CG']
        w = ['GU', 'UG']

        for i in range(len(seq1)):
            ss = seq1[i] + seq2[i]
            if ss in c:
                count_c += 1
            elif ss in w:
                count_w += 1
            else:
                count_mismatch+=1
        result = {'count_c': count_c, 'count_w': count_w, 'count_mismatch' : count_mismatch}
        return result

    def seed_match_type(self):  # 26
        c4 = ['AU', 'UA', 'GC', 'CG']
        w2 = ['GU', 'UG']
        smt_dic = {'Seed_match_8mer': 0,
                   'Seed_match_8merA1': 0,
                   'Seed_match_7mer1': 0,
                   'Seed_match_7mer2': 0,
                   'Seed_match_7merA1': 0,
                   'Seed_match_6mer1': 0,
                   'Seed_match_6mer2': 0,
                   'Seed_match_6mer3': 0,
                   'Seed_match_6mer1GU1': 0,
                   'Seed_match_6mer2GU1': 0,
                   'Seed_match_6mer3GU1': 0,
                   'Seed_match_6mer1GU2': 0,
                   'Seed_match_6mer2GU2': 0,
                   'Seed_match_6mer3GU2': 0,
                   'Seed_match_6mer1GU3': 0,
                   'Seed_match_6mer2GU3': 0,
                   'Seed_match_6mer3GU3': 0,
                   'Seed_match_6mer1GU4': 0,
                   'Seed_match_6mer2GU4': 0,
                   'Seed_match_6mer3GU4': 0,
                   'Seed_match_6mer1GU5': 0,
                   'Seed_match_6mer2GU5': 0,
                   'Seed_match_6mer3GU5': 0,
                   'Seed_match_6mer1GU6': 0,
                   'Seed_match_6mer2GU6': 0,
                   'Seed_match_6mer3GU6': 0,
                   'Seed_match_6mer_LP' : 0,
                   'Seed_match_6mer_BM': 0,
                   'Seed_match_6mer_BT': 0}


        # mirna = self.seed.mir_inter
        # mrna = self.seed.mrna_inter

        # mirna =''
        # for l, nt in self.seed.mir_iterator():
        #     mirna += nt
        # assert len(mirna)==8, "mirna seed is not 8 nt"
        # mrna = self.seed.site
        # mrna = mrna + (8-len(mrna))*"*"

        mr=''
        mi=''

        for i in range(8):
            if self.seed.mrna_inter[i]!=' ':
                mr+=self.seed.mrna_inter[i]
                mi+=self.seed.mir_inter[i]
                continue
            if self.seed.mir_bulge[i]!=' ' and self.seed.mrna_bulge[i]!=' ':
                mr += self.seed.mrna_bulge[i]
                mi += self.seed.mir_bulge[i]
                continue
            if self.seed.mir_bulge[i]!=' ':
                mi += self.seed.mir_bulge[i]
                mr += '-'
                continue
            mr += self.seed.mrna_bulge[i]
            mi += '-'
        mirna = mi
        mrna = mr
        # print (self.seed)
        # print "mrna: " + mr
        # print "mirna:" + mirna

        # # Seed_match_8mer
        if self.seed_complementary(mirna[0:8], mrna[0:8])['count_c'] == 8:
            smt_dic['Seed_match_8mer'] = 1

        # # Seed_match_8merA1
        if self.seed_complementary(mirna[1:8], mrna[1:8])['count_c'] == 7 and mirna[0] == 'A' and mirna[0] + mrna[
            0] not in c4:
            smt_dic['Seed_match_8merA1'] = 1
        if self.seed_complementary(mirna[1:8], mrna[1:8])['count_c'] == 7 and mrna[0] == 'A' and mirna[0] + mrna[
            0] not in c4:
            smt_dic['Seed_match_8merA1'] = 1

        # # Seed_match_7mer1
        if self.seed_complementary(mirna[0:7], mrna[0:7])['count_c'] == 7:
            smt_dic['Seed_match_7mer1'] = 1

        # # Seed_match_7mer2
        if self.seed_complementary(mirna[1:8], mrna[1:8])['count_c'] == 7:
            smt_dic['Seed_match_7mer2'] = 1

        # # Seed_match_7merA1
        if self.seed_complementary(mirna[1:7], mrna[1:7])['count_c'] == 6 and mirna[0] == 'A' and mirna[-1] + mrna[
            0] not in c4:
            smt_dic['Seed_match_7merA1'] = 1
        if self.seed_complementary(mirna[1:7], mrna[1:7])['count_c'] == 6 and mrna[0] == 'A' and mirna[0] + mrna[
            0] not in c4:
            smt_dic['Seed_match_7merA1'] = 1

        # # Seed_match_6mer1, Seed_match_6mer2, Seed_match_6mer3
        if self.seed_complementary(mirna[0:6], mrna[0:6])['count_c'] == 6:
            smt_dic['Seed_match_6mer1'] = 1
        if self.seed_complementary(mirna[1:7], mrna[1:7])['count_c'] == 6:
            smt_dic['Seed_match_6mer2'] = 1
        if self.seed_complementary(mirna[2:8], mrna[2:8])['count_c'] == 6:
            smt_dic['Seed_match_6mer3'] = 1

        ############################################################################################################
        # Code for automatically generating the next section
        #
        # for k in range(1, 4):
        #     print ("#Seed_match_6mer{}GU1,2,3,4,5,6".format(k))
        #     for i in range(5, -1, -1):
        #         print (
        #             "if self.seed_complementary(mirna[{f}:{e}], mrna[{f}:{e}])['count_c'] == {c} and self.seed_complementary(mirna[{f}:{e}], mrna[{f}:{e}])['count_w'] == {w}:".format(
        #                 c=i, w=6 - i, f=k - 1, e=5 + k))
        #         print ("\tsmt_dic['Seed_match_6mer{}GU{}'] = 1".format(k, 6 - i))
        #     print
        ############################################################################################################

        # Seed_match_6mer1GU1,2,3,4,5,6
        if self.seed_complementary(mirna[0:6], mrna[0:6])['count_c'] == 5 and \
                self.seed_complementary(mirna[0:6], mrna[0:6])['count_w'] == 1:
            smt_dic['Seed_match_6mer1GU1'] = 1
        if self.seed_complementary(mirna[0:6], mrna[0:6])['count_c'] == 4 and \
                self.seed_complementary(mirna[0:6], mrna[0:6])['count_w'] == 2:
            smt_dic['Seed_match_6mer1GU2'] = 1
        if self.seed_complementary(mirna[0:6], mrna[0:6])['count_c'] == 3 and \
                self.seed_complementary(mirna[0:6], mrna[0:6])['count_w'] == 3:
            smt_dic['Seed_match_6mer1GU3'] = 1
        if self.seed_complementary(mirna[0:6], mrna[0:6])['count_c'] == 2 and \
                self.seed_complementary(mirna[0:6], mrna[0:6])['count_w'] == 4:
            smt_dic['Seed_match_6mer1GU4'] = 1
        if self.seed_complementary(mirna[0:6], mrna[0:6])['count_c'] == 1 and \
                self.seed_complementary(mirna[0:6], mrna[0:6])['count_w'] == 5:
            smt_dic['Seed_match_6mer1GU5'] = 1
        if self.seed_complementary(mirna[0:6], mrna[0:6])['count_c'] == 0 and \
                self.seed_complementary(mirna[0:6], mrna[0:6])['count_w'] == 6:
            smt_dic['Seed_match_6mer1GU6'] = 1

        # Seed_match_6mer2GU1,2,3,4,5,6
        if self.seed_complementary(mirna[1:7], mrna[1:7])['count_c'] == 5 and \
                self.seed_complementary(mirna[1:7], mrna[1:7])['count_w'] == 1:
            smt_dic['Seed_match_6mer2GU1'] = 1
        if self.seed_complementary(mirna[1:7], mrna[1:7])['count_c'] == 4 and \
                self.seed_complementary(mirna[1:7], mrna[1:7])['count_w'] == 2:
            smt_dic['Seed_match_6mer2GU2'] = 1
        if self.seed_complementary(mirna[1:7], mrna[1:7])['count_c'] == 3 and \
                self.seed_complementary(mirna[1:7], mrna[1:7])['count_w'] == 3:
            smt_dic['Seed_match_6mer2GU3'] = 1
        if self.seed_complementary(mirna[1:7], mrna[1:7])['count_c'] == 2 and \
                self.seed_complementary(mirna[1:7], mrna[1:7])['count_w'] == 4:
            smt_dic['Seed_match_6mer2GU4'] = 1
        if self.seed_complementary(mirna[1:7], mrna[1:7])['count_c'] == 1 and \
                self.seed_complementary(mirna[1:7], mrna[1:7])['count_w'] == 5:
            smt_dic['Seed_match_6mer2GU5'] = 1
        if self.seed_complementary(mirna[1:7], mrna[1:7])['count_c'] == 0 and \
                self.seed_complementary(mirna[1:7], mrna[1:7])['count_w'] == 6:
            smt_dic['Seed_match_6mer2GU6'] = 1

        # Seed_match_6mer3GU1,2,3,4,5,6
        if self.seed_complementary(mirna[2:8], mrna[2:8])['count_c'] == 5 and \
                self.seed_complementary(mirna[2:8], mrna[2:8])['count_w'] == 1:
            smt_dic['Seed_match_6mer3GU1'] = 1
        if self.seed_complementary(mirna[2:8], mrna[2:8])['count_c'] == 4 and \
                self.seed_complementary(mirna[2:8], mrna[2:8])['count_w'] == 2:
            smt_dic['Seed_match_6mer3GU2'] = 1
        if self.seed_complementary(mirna[2:8], mrna[2:8])['count_c'] == 3 and \
                self.seed_complementary(mirna[2:8], mrna[2:8])['count_w'] == 3:
            smt_dic['Seed_match_6mer3GU3'] = 1
        if self.seed_complementary(mirna[2:8], mrna[2:8])['count_c'] == 2 and \
                self.seed_complementary(mirna[2:8], mrna[2:8])['count_w'] == 4:
            smt_dic['Seed_match_6mer3GU4'] = 1
        if self.seed_complementary(mirna[2:8], mrna[2:8])['count_c'] == 1 and \
                self.seed_complementary(mirna[2:8], mrna[2:8])['count_w'] == 5:
            smt_dic['Seed_match_6mer3GU5'] = 1
        if self.seed_complementary(mirna[2:8], mrna[2:8])['count_c'] == 0 and \
                self.seed_complementary(mirna[2:8], mrna[2:8])['count_w'] == 6:
            smt_dic['Seed_match_6mer3GU6'] = 1

        ############################################################################################################
        # Code for automatically generating the next section
        #
        # for k in range(1, 3):
        #     print ("#Seed_match_6mer{}mismatch".format(k))
        #     print (
        #         "if self.seed_complementary(mirna[{f}:{e}], mrna[{f}:{e}])['count_c'] == 6 and self.seed_complementary(mirna[{f}:{e}], mrna[{f}:{e}])['count_mismatch'] == 1:".format(
        #             f=k - 1, e=6 + k))
        #     print ("\tsmt_dic['Seed_match_6mer{}mismatch'] = 1".format(k))
        #     print
        ############################################################################################################


        # Seed_match_6mer_mismatch
        #  LP, BM and BT allow one mismatch. LP has one loop, BM has one bulge on miRNA, and BT has one bulge on the target.
        # Source publication
        # TBD: check with Isana: mirna[0] + mrna[0] not in c4:
        if self.seed_complementary(mirna[1:8], mrna[1:8])['count_c'] == 6 and \
                self.seed_complementary(mirna[1:8], mrna[1:8])['count_mismatch'] == 1  \
                and mrna[0] == 'A':
            mir_buldge = mirna[1:8].find("-")
            mrna_buldge = mrna[1:8].find("-")
            if (mir_buldge==mrna_buldge) :
                smt_dic['Seed_match_6mer_LP'] = 1
            elif mir_buldge != -1 :
                smt_dic['Seed_match_6mer_BT'] = 1
            else :
                smt_dic['Seed_match_6mer_BM'] = 1

           # if self.seed_complementary(mirna[1:8], mrna[1:8])['count_c'] == 6 and \
           #      self.seed_complementary(mirna[1:8], mrna[1:8])['count_mismatch'] == 1  \
           #      and mrna[0] == 'A' and mirna[0] + mrna[0] not in c4:
           #  smt_dic['Seed_match_6mer_mismatch'] = 1


        ###############################################################
        # Update the dict
        ###############################################################
        self.smt_dic = smt_dic
        self.seed_type = [seed_type for seed_type, onehot in smt_dic.items() if onehot == 1]

    def tostring(self):
        classstr = ""
        classstr = classstr + "is canonic : {}\n".format(self.canonic)
        classstr = classstr + "seed type :  {}\n".format(self.seed_type)

        return classstr


    def __str__(self):
        return self.tostring()


def test_seed (seed, seed_type):
    print ("**************************************************")
    print ("Test: " + seed_type)
    s = SeedFeatures(seed)
    s.extract_seed_features()
    print(s.seed_type)
    assert seed_type in s.seed_type, "test error"


def main ():

    s = InteractionRichPresentation("A       ", " GGGGGGG", " CCCCCCC", "C       ")
    test_seed(s, "Seed_match_8merA1")

    s = InteractionRichPresentation("A      C", " GGGGGG ", " CCCCCC ", "C      C")
    test_seed(s, "Seed_match_7merA1")

    s = InteractionRichPresentation("C       ", " GGGGGGG", " CCCCCCC", "C       ")
    test_seed(s, "Seed_match_7mer2")

    s = InteractionRichPresentation("C      C", " GGGGGG ", " CCCCCC ", "C      C")
    test_seed(s, "Seed_match_6mer2")

    s = InteractionRichPresentation("A       ", " GGGGGGG", " CCCCUCC", "C       ")
    test_seed(s, "Seed_match_6mer2GU1")

    s = InteractionRichPresentation("A       ", " GGGGUGG", " CCCCGCC", "C       ")
    test_seed(s, "Seed_match_6mer2GU1")

    s = InteractionRichPresentation("A    C  ", " GGGG GG", " CCCC CC", "C    C  ")
    test_seed(s, "Seed_match_6mer_LP")

    s = InteractionRichPresentation("A       ", " GGGG GG", " CCCC CC", "C    C  ")
    test_seed(s, "Seed_match_6mer_BM")

    s = InteractionRichPresentation("A C       ", " G GGGGGGG", " C CCCCC", "C      ")
    test_seed(s, "Seed_match_6mer_BT")



if __name__ == "__main__":
    main()

