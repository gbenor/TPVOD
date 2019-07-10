from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast import NCBIXML
import datetime


import os
import numpy as np
import pandas as pd

def create_blast_db(db_fasta, db_title):
    print ("create_blast_db")
    cmd = "makeblastdb -in {fasta} -parse_seqids -dbtype nucl -out {out}".format(fasta=db_fasta,
                                                                                 out=db_title)
    print ("{}  {}".format(datetime.datetime.now, cmd))
    os.system(cmd)

def run_blastn(mrna, db_title, blast_output_filname, tmp_dir):
    mRNA_seq = mrna
    mRNA_name = "mrna_to_find"
    filename = tmp_dir / "mrna_to_find.fasta"
    record = SeqRecord(Seq(mRNA_seq), description=mRNA_name)
    SeqIO.write(record, filename, "fasta")
    #think about add strand

    cline = NcbiblastnCommandline(query=str(filename), db=db_title, evalue=0.001,
                                  strand="plus",
                                  out=str(blast_output_filname), outfmt=6)
        # , max_hsps=1,
        #                           max_target_seqs=1)  # it should return the best result
    cmd = str(cline)
    print ("{}  {}".format(datetime.datetime.now(), cmd))
    os.system(cmd)

#
# def parse_blast(blast_result, blast_db_fasta, query_seq):
#     colnames = ['query acc.ver', 'subject acc.ver', '%identity', 'alignment length', 'mismatches', 'gap opens', 'q. start', 'q. end', 's. start', 's. end', 'evalue', 'bit score']
#     result = pd.read_csv(blast_result, sep='\t', names=colnames, header=None)
#     result['gene'] = result['subject acc.ver'].apply(lambda x: x.split('|')[0])
#     if not result['gene'].is_unique:
#         exit (3)
#     full_match_rows = result['%identity'] == 100.0
#
#
#
#     return full_match_rows
#
#
#     E_VALUE_THRESH = 10
#     blast_xml_handle = open(blast_result, "r")
#     records = NCBIXML.parse(blast_xml_handle)
#     for b in records:
#         if b.alignments:  # skip queries with no matches
#             for align in b.alignments:
#                 for hsp in align.hsps:
#                     if hsp.expect < E_VALUE_THRESH:
#                         identities = hsp.identities
#                         full_match = identities == len(query_seq)
#                         sbjct_start = hsp.sbjct_start
#                         sbjct_end = hsp.sbjct_end
#                         title = align.title
#     try:
#         return full_match, title, sbjct_start, sbjct_end, identities
#     except UnboundLocalError:
#         return np.nan, np.nan, np.nan, np.nan, np.nan
