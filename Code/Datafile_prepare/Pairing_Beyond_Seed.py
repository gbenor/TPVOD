from Pipeline import *
from TPVOD_Utils import JsonLog, utils

from Bio import SeqIO

from pathlib import Path
from functools import lru_cache

import pandas as pd
from Bio.SeqFeature import SeqFeature, FeatureLocation
import MirBaseUtils.mirBaseUtils as MBU
import sys, ast


class Pairing_Beyond_Seed(object):
################################################################################
# Strategy:
# Tranform the input file into the form of pipeline.
# Ihe steps are:
# * Extract the target from the genome
# * Extract the miRNA from mature mrna
# When finishing, we can run the pipeline and get the desire datafile.


    def __init__(self, input_file, organism="Celegans", tmp_dir=None, debug=False):
        self.organism = organism
        self.input_file = input_file
        self.tmp_dir = tmp_dir
        self.debug = debug
        self.Pairing_Beyond_Seed_Files()
        self.read_paper_data()

    def Pairing_Beyond_Seed_Files (self):
        self.genome_file = "Wormbase/ce10/Sequence/WholeGenomeFasta/genome.fa"
        self.elegans_with_target = self.tmp_dir / "elegans_with_target.csv"
        self.pairing_beyond_to_pipeline = self.tmp_dir / "pairing_beyond_to_pipeline.csv"
        self.pipeline_output = str(self.tmp_dir / f"{self.organism}_Pairing_Beyond_Seed_pipeline_output.csv")
        self.final_output = str("Datafiles_Prepare/CSV"/  Path (self.organism + "_Pairing_Beyond_Seed_Data.csv"))


    def read_paper_data(self):
        self.sheets = {"elegans" : "Table S2"}
        organism = self.organism.lower()
        if organism == "celegans":
            organism = "elegans"

        current_sheet = self.sheets[organism]

        # Read the interaction file
        inter_df = pd.read_excel(self.input_file, sheet_name=current_sheet, header=None)
        assert "Table S2".lower() in inter_df.iloc[0, 0].lower(), "Read the wrong sheet. no {} in the first cell".format("Table S2")
        if self.debug:
            self.inter_df = pd.read_excel(self.input_file, sheet_name=current_sheet, skiprows=3)[:50]
        else:
            self.inter_df = pd.read_excel(self.input_file, sheet_name=current_sheet, skiprows=3)


    def read_genome (self):
        gen = SeqIO.parse(self.genome_file, "fasta")
        self.genome_list = list (gen)

    @lru_cache(maxsize=None)
    def get_genome_by_id(self, id):
        return [g for g in self.genome_list if g.id == id][0]

    def extract_target (self):
        print("extract_target")
        inter_df = self.inter_df
        #fasta_out, utr_max_len=10000):
        for index, row in inter_df.iterrows():
            chr_id = row['chr']
            chr = self.get_genome_by_id(chr_id)
            strand = row['strand']
            assert strand=='+' or strand=='-', "strand value incorrect {}".format(strand)
            strand = 1 if strand=='+' else -1
            start = row['start']-1 # because python is zero based
            stop = row['stop']
            target_feature = SeqFeature(FeatureLocation(start, stop, strand=strand))
            target = target_feature.location.extract(chr)
            inter_df.loc[index, 'target'] = str(target.seq)
        self.inter_df = inter_df.copy()

    def prepare_for_pipeline(self):
        self.inter_df.rename(
            columns={'mir_ID' : 'microRNA_name',
                     'miRNA_seq' : 'miRNA sequence',
                     'target' : 'target sequence'}	, inplace=True)
        self.inter_df['number of reads'] = 1
        self.inter_df['GI_ID'] = range(len(self.inter_df))
        return self.inter_df

 
    def run(self):
            #####################################################
            # Add the target
            #####################################################
            self.read_genome()
            self.extract_target()

            #####################################################
            # Add the miRNA
            #####################################################
            self.inter_df['mir_ID'] = self.inter_df['name'].apply(lambda x: x[:x.find("_")])
            self.inter_df = MBU.insert_mirna(self.inter_df,  "mir_ID")
            self.inter_df.to_csv(self.elegans_with_target)






def main():
    try:
        debug=ast.literal_eval(sys.argv[1])
    except IndexError:
        debug=True

    if (debug):
        print ("***************************************\n"
               "\t\t\t DEBUG \n"
               "***************************************\n")

    interaction_file = str(Path("Papers/1-s2.0-S1097276516305214-mmc3.xlsx"))
    log_dir = "Datafiles_Prepare/Logs/"
    tmp_dir = utils.make_tmp_dir("Datafiles_Prepare/tmp_dir", parents=True)


    organisms = ["Celegans"]
    for organism in organisms:
        JsonLog.set_filename(
            utils.filename_date_append(Path(log_dir) / Path("Pairing_Beyond_Seed_" + organism + ".json")))
        JsonLog.add_to_json('file name', interaction_file)
        JsonLog.add_to_json('paper',
                            "Pairing beyond the Seed Supports MicroRNA Targeting Specificity")
        JsonLog.add_to_json('Organism', organism)
        JsonLog.add_to_json('paper_url', "https://www.sciencedirect.com/science/article/pii/S1097276516305214#mmc3")

        ce = Pairing_Beyond_Seed(input_file=interaction_file,
                                 organism=organism,
                                 tmp_dir=tmp_dir,
                                 debug=debug)
        ce.run()

        p = Pipeline(paper_name="Pairing_Beyond_Seed",
                     organism=organism,
                     in_df=ce.prepare_for_pipeline(),
                     tmp_dir=tmp_dir)
        p.run()




if __name__ == "__main__":
    main()


