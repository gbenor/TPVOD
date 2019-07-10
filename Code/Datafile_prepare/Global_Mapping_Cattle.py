from Pipeline import *

from TPVOD_Utils import JsonLog, utils

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast import NCBIXML
from pathlib import Path
from functools import lru_cache
import datetime
import os
import pandas as pd
import numpy as np
from Bio.SeqFeature import SeqFeature, FeatureLocation
import MirBaseUtils.mirBaseUtils as MBU

COW_MIRNA_LEN = 22

class Global_Mapping_Cattle(object):
################################################################################
# Strategy:
# Tranform the input file into the form of pipeline.
# Ihe steps are:
# * Extract the target from the genome
# * Extract the miRNA from mature mrna
# When finishing, we can run pipeline and get the desire datafile.


    def __init__(self, input_file, tmp_dir, debug=False):
        organism = "Cow"

        print ("Global mapping of miRNA-target interactions in cattle (Bos taurus)")
        print ("https://www.nature.com/articles/s41598-017-07880-8#MOESM1")
        print("#############################################")
        print("Organism: {}".format(organism))
        print("#############################################")

        self.organism = organism
        self.input_file = input_file
        self.tmp_dir = tmp_dir
        self.debug = debug

        self.Global_Mapping_Cattle_Files()
        self.read_paper_data()

    def Global_Mapping_Cattle_Files (self):
        self.mapping_cattle_to_pipeline = self.tmp_dir / "mapping_cattle__to_pipeline.csv"
        self.final_output = str("Datafiles_Prepare/CSV"/  Path (self.organism + "_Global_Mapping_Cattle_Data.csv"))


    def read_paper_data(self):
        if self.debug:
            self.inter_df = pd.read_csv(self.input_file, nrows=500)
        else:
            self.inter_df = pd.read_csv(self.input_file)
        JsonLog.add_to_json('Num of samples', self.inter_df.shape[0])


    def add_mirna_from_chimera (self):
        def split_seq_to_mirna_target(row):
            seq = str(row['Seq'])
            mirna = seq.split(row['targetSeq'])[0]
            start_idx = row['miRstart'] -1
            return mirna [start_idx:start_idx+COW_MIRNA_LEN]

        def row_excution (row):
            if row['miRNA_seq']!="No mrna match!!!":
                return row['miRNA_seq']
            return split_seq_to_mirna_target(row)

        self.inter_df['miRNA_seq'] = self.inter_df.apply(row_excution, axis=1)



    def prepare_for_pipeline(self):
        self.inter_df.rename(
            columns={'miRNA' : 'microRNA_name',
                     'miRNA_seq' : 'miRNA sequence',
                     'targetSeq' : 'target sequence'}	, inplace=True)
        self.inter_df['number of reads'] = 1
        self.inter_df['GI_ID'] = range(len(self.inter_df))
        return self.inter_df


    def run(self):

        #####################################################
        # Add the miRNA
        # we won't find matches for all of them. we will handle the rest later on.
        #####################################################
        self.inter_df = MBU.insert_mirna(self.inter_df, "miRNA", "bta")

        #####################################################
        # Add the miRNA from the chimera
        #####################################################
        self.add_mirna_from_chimera()
        self.inter_df['miRNA_seq'] = self.inter_df.apply(lambda row: row['miRNA_seq'].replace("U", "T"), axis=1)



def main ():
    debug = False
    interaction_file = str(Path("Papers/41598_2017_7880_MOESM4_ESM.csv"))
    log_dir = "Datafiles_Prepare/Logs/"
    tmp_dir = utils.make_tmp_dir("Datafiles_Prepare/tmp_dir", parents=True)

    organisms = ["Cow"]
    for organism in organisms:
        JsonLog.set_filename(
            utils.filename_date_append(Path(log_dir) / Path("Global_Mapping_Cattle_" + organism + ".json")))
        JsonLog.add_to_json('file name', interaction_file)
        JsonLog.add_to_json('paper',
                            "Global mapping of miRNA-target interactions in cattle (Bos taurus)")
        JsonLog.add_to_json('Organism', organism)
        JsonLog.add_to_json('paper_url', "https://www.nature.com/articles/s41598-017-07880-8#MOESM1")

        cow = Global_Mapping_Cattle(input_file=interaction_file,
                                    tmp_dir=tmp_dir,
                                    debug=debug)

        cow.run()

        p = Pipeline(paper_name="Global_Mapping_Cattle",
                     organism=organism,
                     in_df=cow.prepare_for_pipeline(),
                     tmp_dir=tmp_dir)
        p.run()




if __name__ == "__main__":
    main()



