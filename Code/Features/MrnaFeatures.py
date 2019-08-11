import pandas as pd
from functools import reduce

class MrnaFeatures(object):

    def __init__(self, mir, site, site_start_loc, site_end_loc, full_mrna, flank_number=70, hot_encoding_len=9):
        self.mir = mir
        self.site = site
        self.site_start_loc = site_start_loc
        self.site_end_loc = site_end_loc
        self.full_mrna = full_mrna
        self.flank_number = flank_number
        self.hot_encoding_len = hot_encoding_len


    def extract_mrna_features(self):
        mrna = self.full_mrna
        mr_site = self.site
        mr_site_loc = [self.site_start_loc, self.site_end_loc]

        self.dte = self.distance_to_end(mrna, mr_site_loc)
        self.dts = self.distance_to_start(mrna, mr_site_loc)

        self.tc = self.target_composition(mr_site)
        self.fuc = self.flanking_up_composition(mrna, mr_site_loc, flank_number=self.flank_number)
        self.fdc = self.flanking_down_composition(mrna, mr_site_loc, flank_number=self.flank_number)
        self.PHE = self.pair_hot_encoding(self.mir, mr_site, self.hot_encoding_len)

    def get_features(self):
        f = [self.dts, self.dte, self.tc, self.fuc, self.fdc, self.PHE]
        df = map(lambda x: pd.DataFrame([x]), f)
        r =  reduce (lambda x,y: pd.concat([x, y], axis=1, sort=False), df)
        return r




    # # 3. location of target site (1)
    def distance_to_end(self, mrna, mr_site_loc):  # 1
        dte_dic = {'MRNA_Dist_to_end': round(float(len(mrna) - int(mr_site_loc[1])) / len(mrna), 4)}
        return dte_dic

    def distance_to_start(self, mrna, mr_site_loc):  # 1
        dts_dic = {'MRNA_Dist_to_start': round(float(mr_site_loc[0]) / len(mrna), 4)}
        #float(mr_site_loc[0]) / len(mrna)
        return dts_dic

    # # 4. target composition (20+20+20)
    def target_composition(self, mr_site):  # 20
        mrna = mr_site.upper().replace('T', 'U')
        count_A = 0
        count_U = 0
        count_G = 0
        count_C = 0
        count_AA = 0
        count_AU = 0
        count_AG = 0
        count_AC = 0
        count_UA = 0
        count_UU = 0
        count_UG = 0
        count_UC = 0
        count_GA = 0
        count_GU = 0
        count_GG = 0
        count_GC = 0
        count_CA = 0
        count_CU = 0
        count_CG = 0
        count_CC = 0
        for i in range(len(mrna)):
            if mrna[i] == 'A':
                count_A += 1
            elif mrna[i] == 'U':
                count_U += 1
            elif mrna[i] == 'G':
                count_G += 1
            elif mrna[i] == 'C':
                count_C += 1
        for i in range(len(mrna) - 1):
            if mrna[i:i + 2] == 'AA':
                count_AA += 1
            elif mrna[i:i + 2] == 'AU':
                count_AU += 1
            elif mrna[i:i + 2] == 'AG':
                count_AG += 1
            elif mrna[i:i + 2] == 'AC':
                count_AC += 1
            elif mrna[i:i + 2] == 'UA':
                count_UA += 1
            elif mrna[i:i + 2] == 'UU':
                count_UU += 1
            elif mrna[i:i + 2] == 'UG':
                count_UG += 1
            elif mrna[i:i + 2] == 'UC':
                count_UC += 1
            elif mrna[i:i + 2] == 'GA':
                count_GA += 1
            elif mrna[i:i + 2] == 'GU':
                count_GU += 1
            elif mrna[i:i + 2] == 'GG':
                count_GG += 1
            elif mrna[i:i + 2] == 'GC':
                count_GC += 1
            elif mrna[i:i + 2] == 'CA':
                count_CA += 1
            elif mrna[i:i + 2] == 'CU':
                count_CU += 1
            elif mrna[i:i + 2] == 'CG':
                count_CG += 1
            elif mrna[i:i + 2] == 'CC':
                count_CC += 1

        all_monomer_count = count_A + count_U + count_G + count_C
        all_dimer_count = count_AA + count_AU + count_AG + count_AC + \
                          count_UA + count_UU + count_UG + count_UC + \
                          count_GA + count_GU + count_GG + count_GC + \
                          count_CA + count_CU + count_CG + count_CC
        tc_dic = {'MRNA_Target_A_comp': round(float(count_A) / all_monomer_count, 4),
                  'MRNA_Target_U_comp': round(float(count_U) / all_monomer_count, 4),
                  'MRNA_Target_G_comp': round(float(count_G) / all_monomer_count, 4),
                  'MRNA_Target_C_comp': round(float(count_C) / all_monomer_count, 4),
                  'MRNA_Target_AA_comp': round(float(count_AA) / all_dimer_count, 4),
                  'MRNA_Target_AU_comp': round(float(count_AU) / all_dimer_count, 4),
                  'MRNA_Target_AG_comp': round(float(count_AG) / all_dimer_count, 4),
                  'MRNA_Target_AC_comp': round(float(count_AC) / all_dimer_count, 4),
                  'MRNA_Target_UA_comp': round(float(count_UA) / all_dimer_count, 4),
                  'MRNA_Target_UU_comp': round(float(count_UU) / all_dimer_count, 4),
                  'MRNA_Target_UG_comp': round(float(count_UG) / all_dimer_count, 4),
                  'MRNA_Target_UC_comp': round(float(count_UC) / all_dimer_count, 4),
                  'MRNA_Target_GA_comp': round(float(count_GA) / all_dimer_count, 4),
                  'MRNA_Target_GU_comp': round(float(count_GU) / all_dimer_count, 4),
                  'MRNA_Target_GG_comp': round(float(count_GG) / all_dimer_count, 4),
                  'MRNA_Target_GC_comp': round(float(count_GC) / all_dimer_count, 4),
                  'MRNA_Target_CA_comp': round(float(count_CA) / all_dimer_count, 4),
                  'MRNA_Target_CU_comp': round(float(count_CU) / all_dimer_count, 4),
                  'MRNA_Target_CG_comp': round(float(count_CG) / all_dimer_count, 4),
                  'MRNA_Target_CC_comp': round(float(count_CC) / all_dimer_count, 4)}
        return tc_dic

    def flanking_up_composition(self, mrna, mr_site_loc, flank_number=70):  # 20
        mrna_full = mrna.upper().replace('T', 'U')
        mrna_up = mrna_full[max(0, mr_site_loc[0] - 70):mr_site_loc[0]]
        mrna_down = mrna_full[mr_site_loc[1] + 1:mr_site_loc[1] + 71]
        # print len(mrna_up), len(mrna_down)

        # # Up
        count_A = 0
        count_U = 0
        count_G = 0
        count_C = 0
        count_AA = 0
        count_AU = 0
        count_AG = 0
        count_AC = 0
        count_UA = 0
        count_UU = 0
        count_UG = 0
        count_UC = 0
        count_GA = 0
        count_GU = 0
        count_GG = 0
        count_GC = 0
        count_CA = 0
        count_CU = 0
        count_CG = 0
        count_CC = 0
        for i in range(len(mrna_up)):
            if mrna_up[i] == 'A':
                count_A += 1
            elif mrna_up[i] == 'U':
                count_U += 1
            elif mrna_up[i] == 'G':
                count_G += 1
            elif mrna_up[i] == 'C':
                count_C += 1
        for i in range(len(mrna_up) - 1):
            if mrna_up[i:i + 2] == 'AA':
                count_AA += 1
            elif mrna_up[i:i + 2] == 'AU':
                count_AU += 1
            elif mrna_up[i:i + 2] == 'AG':
                count_AG += 1
            elif mrna_up[i:i + 2] == 'AC':
                count_AC += 1
            elif mrna_up[i:i + 2] == 'UA':
                count_UA += 1
            elif mrna_up[i:i + 2] == 'UU':
                count_UU += 1
            elif mrna_up[i:i + 2] == 'UG':
                count_UG += 1
            elif mrna_up[i:i + 2] == 'UC':
                count_UC += 1
            elif mrna_up[i:i + 2] == 'GA':
                count_GA += 1
            elif mrna_up[i:i + 2] == 'GU':
                count_GU += 1
            elif mrna_up[i:i + 2] == 'GG':
                count_GG += 1
            elif mrna_up[i:i + 2] == 'GC':
                count_GC += 1
            elif mrna_up[i:i + 2] == 'CA':
                count_CA += 1
            elif mrna_up[i:i + 2] == 'CU':
                count_CU += 1
            elif mrna_up[i:i + 2] == 'CG':
                count_CG += 1
            elif mrna_up[i:i + 2] == 'CC':
                count_CC += 1
        all_monomer_count = count_A + count_U + count_G + count_C
        all_dimer_count = count_AA + count_AU + count_AG + count_AC + \
                          count_UA + count_UU + count_UG + count_UC + \
                          count_GA + count_GU + count_GG + count_GC + \
                          count_CA + count_CU + count_CG + count_CC
        if all_monomer_count == 0:
            all_monomer_count += 70
        if all_dimer_count == 0:
            all_dimer_count += 70
        fuc_dic = {'MRNA_Up_A_comp': round(float(count_A) / all_monomer_count, 4),
                   'MRNA_Up_U_comp': round(float(count_U) / all_monomer_count, 4),
                   'MRNA_Up_G_comp': round(float(count_G) / all_monomer_count, 4),
                   'MRNA_Up_C_comp': round(float(count_C) / all_monomer_count, 4),
                   'MRNA_Up_AA_comp': round(float(count_AA) / all_dimer_count, 4),
                   'MRNA_Up_AU_comp': round(float(count_AU) / all_dimer_count, 4),
                   'MRNA_Up_AG_comp': round(float(count_AG) / all_dimer_count, 4),
                   'MRNA_Up_AC_comp': round(float(count_AC) / all_dimer_count, 4),
                   'MRNA_Up_UA_comp': round(float(count_UA) / all_dimer_count, 4),
                   'MRNA_Up_UU_comp': round(float(count_UU) / all_dimer_count, 4),
                   'MRNA_Up_UG_comp': round(float(count_UG) / all_dimer_count, 4),
                   'MRNA_Up_UC_comp': round(float(count_UC) / all_dimer_count, 4),
                   'MRNA_Up_GA_comp': round(float(count_GA) / all_dimer_count, 4),
                   'MRNA_Up_GU_comp': round(float(count_GU) / all_dimer_count, 4),
                   'MRNA_Up_GG_comp': round(float(count_GG) / all_dimer_count, 4),
                   'MRNA_Up_GC_comp': round(float(count_GC) / all_dimer_count, 4),
                   'MRNA_Up_CA_comp': round(float(count_CA) / all_dimer_count, 4),
                   'MRNA_Up_CU_comp': round(float(count_CU) / all_dimer_count, 4),
                   'MRNA_Up_CG_comp': round(float(count_CG) / all_dimer_count, 4),
                   'MRNA_Up_CC_comp': round(float(count_CC) / all_dimer_count, 4)}
        return fuc_dic

    def flanking_down_composition(self, mrna, mr_site_loc, flank_number=70):  # 20
        mrna_full = mrna.upper().replace('T', 'U')
        mrna_up = mrna_full[max(0, mr_site_loc[0] - 70):mr_site_loc[0]]
        mrna_down = mrna_full[mr_site_loc[1] + 1:mr_site_loc[1] + 71]
        # print len(mrna_up), len(mrna_down)

        # # Down
        count_A = 0
        count_U = 0
        count_G = 0
        count_C = 0
        count_AA = 0
        count_AU = 0
        count_AG = 0
        count_AC = 0
        count_UA = 0
        count_UU = 0
        count_UG = 0
        count_UC = 0
        count_GA = 0
        count_GU = 0
        count_GG = 0
        count_GC = 0
        count_CA = 0
        count_CU = 0
        count_CG = 0
        count_CC = 0
        for i in range(len(mrna_down)):
            if mrna_down[i] == 'A':
                count_A += 1
            elif mrna_down[i] == 'U':
                count_U += 1
            elif mrna_down[i] == 'G':
                count_G += 1
            elif mrna_down[i] == 'C':
                count_C += 1
        for i in range(len(mrna_down) - 1):
            if mrna_down[i:i + 2] == 'AA':
                count_AA += 1
            elif mrna_down[i:i + 2] == 'AU':
                count_AU += 1
            elif mrna_down[i:i + 2] == 'AG':
                count_AG += 1
            elif mrna_down[i:i + 2] == 'AC':
                count_AC += 1
            elif mrna_down[i:i + 2] == 'UA':
                count_UA += 1
            elif mrna_down[i:i + 2] == 'UU':
                count_UU += 1
            elif mrna_down[i:i + 2] == 'UG':
                count_UG += 1
            elif mrna_down[i:i + 2] == 'UC':
                count_UC += 1
            elif mrna_down[i:i + 2] == 'GA':
                count_GA += 1
            elif mrna_down[i:i + 2] == 'GU':
                count_GU += 1
            elif mrna_down[i:i + 2] == 'GG':
                count_GG += 1
            elif mrna_down[i:i + 2] == 'GC':
                count_GC += 1
            elif mrna_down[i:i + 2] == 'CA':
                count_CA += 1
            elif mrna_down[i:i + 2] == 'CU':
                count_CU += 1
            elif mrna_down[i:i + 2] == 'CG':
                count_CG += 1
            elif mrna_down[i:i + 2] == 'CC':
                count_CC += 1
        all_monomer_count = count_A + count_U + count_G + count_C
        all_dimer_count = count_AA + count_AU + count_AG + count_AC + \
                          count_UA + count_UU + count_UG + count_UC + \
                          count_GA + count_GU + count_GG + count_GC + \
                          count_CA + count_CU + count_CG + count_CC
        if all_monomer_count == 0:
            all_monomer_count += 70
        if all_dimer_count == 0:
            all_dimer_count += 70
        fdc_dic = {'MRNA_Down_A_comp': round(float(count_A) / all_monomer_count, 4),
                   'MRNA_Down_U_comp': round(float(count_U) / all_monomer_count, 4),
                   'MRNA_Down_G_comp': round(float(count_G) / all_monomer_count, 4),
                   'MRNA_Down_C_comp': round(float(count_C) / all_monomer_count, 4),
                   'MRNA_Down_AA_comp': round(float(count_AA) / all_dimer_count, 4),
                   'MRNA_Down_AU_comp': round(float(count_AU) / all_dimer_count, 4),
                   'MRNA_Down_AG_comp': round(float(count_AG) / all_dimer_count, 4),
                   'MRNA_Down_AC_comp': round(float(count_AC) / all_dimer_count, 4),
                   'MRNA_Down_UA_comp': round(float(count_UA) / all_dimer_count, 4),
                   'MRNA_Down_UU_comp': round(float(count_UU) / all_dimer_count, 4),
                   'MRNA_Down_UG_comp': round(float(count_UG) / all_dimer_count, 4),
                   'MRNA_Down_UC_comp': round(float(count_UC) / all_dimer_count, 4),
                   'MRNA_Down_GA_comp': round(float(count_GA) / all_dimer_count, 4),
                   'MRNA_Down_GU_comp': round(float(count_GU) / all_dimer_count, 4),
                   'MRNA_Down_GG_comp': round(float(count_GG) / all_dimer_count, 4),
                   'MRNA_Down_GC_comp': round(float(count_GC) / all_dimer_count, 4),
                   'MRNA_Down_CA_comp': round(float(count_CA) / all_dimer_count, 4),
                   'MRNA_Down_CU_comp': round(float(count_CU) / all_dimer_count, 4),
                   'MRNA_Down_CG_comp': round(float(count_CG) / all_dimer_count, 4),
                   'MRNA_Down_CC_comp': round(float(count_CC) / all_dimer_count, 4)}
        return fdc_dic

    # # 8. pair hot-encoding
    def pair_hot_encoding(self, mir, mr_site, length):
        mirna = mir.upper().replace('T', 'U')[:length]
        mrna = mr_site.upper().replace('T', 'U')[:length]

        def hot_coding(seq):
            if seq == 'A' or seq == 'a':
                he = [1, 0, 0, 0, 0]
            elif seq == 'U' or seq == 'u':
                he = [0, 1, 0, 0, 0]
            elif seq == 'T' or seq == 't':
                he = [0, 1, 0, 0, 0]
            elif seq == 'G' or seq == 'g':
                he = [0, 0, 1, 0, 0]
            elif seq == 'C' or seq == 'c':
                he = [0, 0, 0, 1, 0]
            else:
                he = [0, 0, 0, 0, 1]
            return he

        PHE = {}
        for i in range(len(mirna)):
            for j in range(5):
                key = 'HotPairingMirna_he_P%s_L%s' % (str(i + 1), str(j + 1))
                PHE[key] = hot_coding(mirna[i])[j]

        for i in range(len(mrna)):
            for j in range(5):
                key = 'HotPairingMRNA_he_P%s_L%s' % (str(i + 1), str(j + 1))
                PHE[key] = hot_coding(mrna[i])[j]
        return PHE

    # def tostring(self):
    #     # mmp = pd.DataFrame([self.mmp_dic])
    #     # mpc = pd.DataFrame([self.mpc_dic])
    #     # pd.set_option('display.max_columns', None)
    #     #
    #     # classstr = ""
    #     #
    #     # classstr = classstr + str(mmp) + "\n"
    #     # classstr = classstr + str(mpc) + "\n"
    #     #
    #     # return classstr
    #
    #
    # def __str__(self):
    #     return self.tostring()
