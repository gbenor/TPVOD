import BlastUtils
from pathlib import Path
import JsonLog
from Bio import SeqIO
import datetime
import pandas as pd
from datetime import datetime
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast import NCBIXML

import os
import numpy as np
import pandas as pd
def run_blastn(mrna, db_title, blast_output_filname):
    mRNA_seq = mrna
    mRNA_name = "mrna_to_find"
    filename = "mrna_to_find.fasta"
    record = SeqRecord(Seq(mRNA_seq), description=mRNA_name)
    SeqIO.write(record, filename, "fasta")
    #think about add strand

    cline = NcbiblastnCommandline(query=filename, db=db_title, evalue=0.001,
                                  strand="plus",
                                  out=blast_output_filname, outfmt=6)
    cmd = str(cline)
    print (cmd)
    os.system(cmd)

input_dir = Path("Datafiles_Prepare/Logs/")
files = list(input_dir.iterdir())
files = [x for x in files if x.suffix==".fasta"]
blast_tmp_file = "blast_exc.csv"
for f in files:
    print ("#####################################################")
    with open(f.stem +".txt", "w") as text_file:
        org = f.stem.split("_")[0]
        print (org)
        p_dir = Path(org) / "Raw"
        blast_db_biomart = str(p_dir / "blast_files" / "blastdb_biomart")

        for record in SeqIO.parse(f, "fasta"):
            tar_seq = str(record.seq)
            if tar_seq == "TAAAAATGGTGGCAACATCATCTCGTTGGTAGGAATTTTTTACTTGAATTGTTATTTT":
                print("eta")

            run_blastn(tar_seq, blast_db_biomart, blast_tmp_file)
            text_file.write ("\n\n\n##################################################\n")
            text_file.write(tar_seq)
            text_file.write("\n")
            non_unique_seq = []

            colnames = ['query acc.ver', 'subject acc.ver', '%identity', 'alignment length', 'mismatches',
                        'gap opens', 'q.start', 'q.end', 's.start', 's.end', 'evalue', 'bit score']

            result = pd.read_csv(blast_tmp_file, sep='\t', names=colnames, header=None)

            # Consider the full match rows only
            #####################################
            full_match_rows = result['%identity'] == 100.0
            result = result[full_match_rows]
            result.reset_index(drop=True, inplace=True)

            full_match_rows = result['alignment length'] == len(tar_seq)
            result = result[full_match_rows]
            result.reset_index(drop=True, inplace=True)

            full_match_rows = result['gap opens'] == 0
            result = result[full_match_rows]
            result.reset_index(drop=True, inplace=True)

            result = result.filter(
                ['query acc.ver', 'subject acc.ver', 'alignment length', 'mismatches',
                 'gap opens', 's.start', 's.end'], axis=1)

            # text_file.writelines(open("blast_exc.txt", "r").readlines())
            text_file.write(result.to_string())




