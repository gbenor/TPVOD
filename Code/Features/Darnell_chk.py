import pandas as pd
import numpy as np
from SeedFeaturesCompact import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from Bio import SeqIO
from collections import Counter
import RNA
from ViennaRNADuplex import *
from SeedFeatures import *
from MatchingFeatures import *
from MirandaDuplex import *
from MrnaFeatures import *
from EnergyAccess import *
from pathlib import Path
from Utils import *
from NegativeSamples import *
import JsonLog
import itertools
from multiprocessing import Pool
import multiprocessing
import pprint
# feature_file = "Data/Features/CSV/Human_Darnell_miRNA_target_chimeras_Data_20190508-102813_20190508-103611_vienna_valid_seeds_all_duplex_20190508-115838.csv"
# darnel_file = "Data/Human/Raw/Human_mRNA_20190508-102813.csv"
import jellyfish
import subprocess


def my_rep (s):
    s = s.replace ("(", "|")
    s = s.replace (")", "|")
    s = s.replace (".", "-")
    s = s.replace ("W", "|")
    s = s + (22-len(s))*"-"

    return s

def interaction_form (s1,s2):
    t = ""
    for i in range(len(s1)):
        if s1[i]!=" ":
            t+="-"
        elif s2[i]!=" ":
            t+="|"

    return t

def rnaHybrid (mrna, mirna, seed_pos):
    cmd = "RNAhybrid -s 3utr_human {} {}".format(mrna, mirna)
    print (cmd)
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    a = p.stdout.readlines()
    b = [[a[l], a[l + 1], a[l + 2], a[l + 3]] for l in range(len(a)) if a[l].startswith('target 5')]
    end = b[0][0].find("3") - 1
    f = [t[10:end] for t in b[0]]
    f = [t[::-1] for t in f]
    tar = interaction_form(f[0], f[1])[1:]
    tar += "-" * max(0, (22 - len(tar)))
    #tar = tar[:22]

    mir = interaction_form(f[3], f[2])[1:]
    mir += "-" * max(0, (22 - len(mir)))
    #mir = mir[:22]
    return tar, mir



feature_file = "Data/Features/CSV/Mouse_Darnell_miRNA_target_chimeras_Data_20190508-102318_20190508-102718_vienna_valid_seeds_all_duplex_20190508-111855.csv"
darnel_file = "Data/Mouse/Raw/Mouse_mRNA_20190508-102318.csv"
#darnel_file = "Data/Human/Raw/Human_mRNA_20190508-102813.csv"

feature_file = darnel_file

feature_df = pd.read_csv(feature_file)
darnel_df = pd.read_csv(darnel_file)

i=0

sum = 0
for index, row in feature_df.iterrows():
    i += 1
    # if i> 1000:
    #     break

    print ("***********************************************************")
    dp = ViennaRNADuplex(row['miRNA sequence'], row['target sequence'])
    #print (dp)
    v_mir = my_rep(dp.duplex.structure.split("&")[0])
    v_mrna = my_rep(dp.duplex.structure.split("&")[1][::-1])





    gi_id = row['GI_ID']
    darnel_row = darnel_df[darnel_df['GI_ID'] == gi_id]
    darnel_row.reset_index(inplace=True)
    darnell_mir = my_rep(darnel_row.loc[0, "miR.map"])
    darnell_mrna = my_rep(darnel_row.loc[0, "target.map"])




    mir_dist =  (jellyfish.levenshtein_distance(v_mir.decode('unicode-escape'), darnell_mir.decode('unicode-escape')))
    mrna_dist =  (jellyfish.levenshtein_distance(v_mrna.decode('unicode-escape'), darnell_mrna.decode('unicode-escape')))

    if row['target sequence'] =="GAAAATCAGAACAGGGTAGACAGCTGTTAAAAACAATGTTTAAATGGAATAATGTTGAATGTTTACAGGCTGTAAG":
        print ("eta")

    tar, mir = rnaHybrid (row['target sequence'], row['miRNA sequence'], row['seed position'])

    print (tar)
    print (darnell_mrna)
    print (jellyfish.levenshtein_distance(tar.decode('unicode-escape'), darnell_mrna.decode('unicode-escape')))

    print ("")
    print (mir)
    print (darnell_mir)
    print (jellyfish.levenshtein_distance(mir.decode('unicode-escape'), darnell_mir.decode('unicode-escape')))


    # for e in b[0]:
    #     print (e[::-1])

    if True:
            #max(mir_dist, mrna_dist) > 4:
        sum +=1

    #     print (v_mir)
    #     print (darnell_mir)
    #     print (mir_dist)
    #     print (v_mrna)
    #     print (darnell_mrna)
    #     print (mrna_dist)
    # print ("Total of {} unmatched duplexes out of {} dups".format(sum, i))
    if sum>10:
        exit(7)



