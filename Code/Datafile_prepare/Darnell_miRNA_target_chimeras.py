from Pipeline import *
from TPVOD_Utils import JsonLog, utils

from Bio import SeqIO

from pathlib import Path
try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache

import pandas as pd
from Bio.SeqFeature import SeqFeature, FeatureLocation
import MirBaseUtils.mirBaseUtils as MBU
import sortedcontainers as SC
import jellyfish
from Duplex import rnaHybrid


class Darnell_miRNA_target_chimeras(object):
    ################################################################################
    # Strategy:
    # Tranform the input file into the form of pipeline.
    # Ihe steps are:
    # * Extract the target from the genome
    # * Extract the miRNA from mature mrna
    # When finishing, we can run the pipeline and get the desire datafile.

    def __init__(self, input_file, tmp_dir, organism, debug=False):
        self.organism = organism.lower()
        self.input_file = input_file
        self.tmp_dir = tmp_dir
        self.debug = debug
        self.Darnell_Files()
        self.read_paper_data()

    def Darnell_Files(self):
        if self.organism == "mouse" :
            self.genome_dir = "Genome/Mouse/mm9/"
        elif self.organism == "human" :
            self.genome_dir = "Genome/Human/hg18/"
        else:
            raise Exception ("Wrong organism: {}".format(self.organism))
        self.pipeline_in_file = str(self.tmp_dir / Path(self.organism + "_Darnell_with_targets.csv"))


    def read_paper_data(self):
        # Read the interaction file
        skiprows = 24 if self.organism=="mouse" else 23
        if self.debug:
            print ("read paper debug")
            self.inter_df = pd.read_excel(self.input_file, skiprows=skiprows)[:500]
        else:
            self.inter_df = pd.read_excel(self.input_file, skiprows=skiprows)


    def select_3utr (self):
        UTR3 = "3\'UTR"
        self.inter_df = self.inter_df[self.inter_df["region"]==UTR3]

    @lru_cache(maxsize=None)
    def get_genome_by_id(self, id):
        fname = "{}/{}.fa".format(self.genome_dir, id)
        gen = SeqIO.parse(fname, "fasta")
        l = list (gen)
        assert len(l)==1, "read genome error. this chr hasn't had exactly one chr. chr={}".format(id)
        return l[0]

    def extract_target(self):
        inter_df = self.inter_df
        i=0
        for index, row in inter_df.iterrows():
            i+=1
            print (i)
            # if i > 20:
            #     break
            chr_id = row['chr']
            chr = self.get_genome_by_id(chr_id)
            strand = row['strand']
            assert strand == '+' or strand == '-', "strand value incorrect {}".format(strand)
            strand = 1 if strand == '+' else -1
            start = row['start']-1 #Note that the start and end location numbering follow Python's scheme
            stop = row['end']
            target_feature = SeqFeature(FeatureLocation(start, stop, strand=strand))
            target = target_feature.location.extract(chr)
            inter_df.loc[index, 'target'] = str(target.seq)
            #print (str(target.seq))
        self.inter_df = inter_df.copy()

    def calc_levenshtein_distance (self, mirna, mrna, mirna_pattern, mrna_pattern):

        tar, mir = rnaHybrid.rnaHybrid(mrna=mrna, mirna=mirna)

        # mir_dist = (
        #     jellyfish.levenshtein_distance(mir.decode('unicode-escape'), mirna_pattern))
        # mrna_dist = (
        #     jellyfish.levenshtein_distance(tar.decode('unicode-escape'), mrna_pattern.decode('unicode-escape')))

        mir_dist = (
            jellyfish.levenshtein_distance(mir, mirna_pattern))
        mrna_dist = (
            jellyfish.levenshtein_distance(tar, mrna_pattern))

        return min(mir_dist, mrna_dist)

    def add_mirna(self, MAX_DISTANCE=4):
        mirna_prefix = None
        if self.organism == "mouse":
            mirna_prefix = "mmu"
        elif self.organism == "human":
            mirna_prefix = "hsa"
        mirbasedic = MBU.read_mirbase_file(mirna_prefix)

        stat_num_of_direct_mirna = 0
        stat_num_of_heuristic_mirna = 0

        for index, row in self.inter_df.iterrows():
            self.inter_df.loc[index, 'miRNA_seq'] = "No mrna match!!!"
            mirna_id = row["miRNA"]
            mrna = row['target']
            darnell_mir_p = row["miR.map"]
            darnell_mrna_p = row["target.map"]

            mirna_list = MBU.find_mirnas_by_prefix(mirbasedic, mirna_id)
            if len(mirna_list) == 1:
                self.inter_df.loc[index, 'miRNA_seq'] = mirna_list[0][1]
                stat_num_of_direct_mirna += 1

            elif len(mirna_list) > 1:
                result_dict = SC.SortedDict()
                for id, seq in mirna_list:
                    dist = self.calc_levenshtein_distance(seq, mrna, darnell_mir_p, darnell_mrna_p)
                    result_dict[(id, seq)] = dist

                v = list(result_dict.values())
                k = list(result_dict.keys())
                min_value = min(v)
                if min_value <= MAX_DISTANCE:
                    m_seq = k[v.index(min_value)][1]
                    print(m_seq)
                    self.inter_df.loc[index, 'miRNA_seq'] = m_seq
                    stat_num_of_heuristic_mirna += 1

        JsonLog.add_to_json("num_of_direct_mirna", stat_num_of_direct_mirna)
        JsonLog.add_to_json("num_of_heuristic_mirna", stat_num_of_heuristic_mirna)




    def prepare_for_pipeline(self):
        self.inter_df.rename(
            columns={'miRNA': 'microRNA_name',
                     'miRNA_seq': 'miRNA sequence',
                     'target': 'target sequence'}, inplace=True)
        self.inter_df['number of reads'] = 1
        self.inter_df['GI_ID'] = range(len(self.inter_df))
        return self.inter_df

    def run(self):
        JsonLog.add_to_json("Samples", self.inter_df.shape[0])
        #####################################################
        # Select only 3'utr
        #####################################################
        self.select_3utr()
        JsonLog.add_to_json("valid_utr3_before", self.inter_df.shape[0])


        #####################################################
        # Add the target
        #####################################################
        self.extract_target()

        #####################################################
        # Add the miRNA
        #####################################################
        self.add_mirna()

        #####################################################
        # Remove "No mrna match!!!"
        #####################################################
        self.inter_df = self.inter_df[self.inter_df["miRNA_seq"]!="No mrna match!!!"]
        JsonLog.add_to_json("valid_mirna_before", self.inter_df.shape[0])


        print ("save target file: {}".format(self.pipeline_in_file))
        self.inter_df.to_csv(self.pipeline_in_file)


def main():
    debug=False
    mouse_config = {"organism" : "Mouse",
                    "interaction_file" : "Papers/ncomms9864-s2.xlsx"}
    human_config = {"organism": "Human",
                    "interaction_file": "Papers/ncomms9864-s4.xlsx"}


    tmp_dir = utils.make_tmp_dir("Datafiles_Prepare/tmp_dir", parents=True)
    log_dir = "Datafiles_Prepare/Logs/"

    for cnfg in [mouse_config, human_config]:
        organism = cnfg["organism"]
        interaction_file = cnfg["interaction_file"]

        JsonLog.set_filename(
            utils.filename_date_append(Path(log_dir) / Path("Darnell_miRNA_target_chimeras_" + organism + ".json")))
        JsonLog.add_to_json('file name', interaction_file)
        JsonLog.add_to_json('paper',"miRNA–target chimeras reveal miRNA 3-end pairing as a major determinant of Argonaute target specificity")
        JsonLog.add_to_json('Organism', organism)
        JsonLog.add_to_json('paper_url', "https://www.nature.com/articles/ncomms9864")

        org = Darnell_miRNA_target_chimeras(interaction_file, tmp_dir, organism, debug=debug)
        org.run()

        print ("Pipeline start")
        p = Pipeline(paper_name="Darnell_miRNA_target_chimeras",
                     organism=organism,
                     in_df=org.prepare_for_pipeline(),
                     tmp_dir=tmp_dir)

        p.run()


if __name__ == "__main__":
    main()


