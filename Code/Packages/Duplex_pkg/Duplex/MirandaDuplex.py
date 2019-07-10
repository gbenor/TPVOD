import pandas as pd
import numpy as np
from Bio.Blast.Applications import NcbiblastnCommandline
from ViennaRNADuplex import *

from InteractionRichPresentation import *

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os
import itertools as it
from itertools import islice


class MirandaDuplex(object):

    def __init__(self, mir, mrna, tmpdir):
        self.mir = mir
        self.run_miranda(mir, mrna, tmpdir)
        self.IRP = self.parse_miranda_output(self.miranda_out_file)




    def run_miranda (self, mir, mrna, tmpdir):
        mrna_fasta_filename = tmpdir + "/tmp_mrna{}.fasta".format(os.getpid())
        mirna_fasta_filename = tmpdir + "/tmp_mirna{}.fasta".format(os.getpid())
        miranda_out = tmpdir + "/miranda_out{}.txt".format(os.getpid())

        mRNA_record = SeqRecord(Seq(mrna), description="mRNA")
        miRNA_record = SeqRecord(Seq(mir), description="miRNA")
        SeqIO.write(mRNA_record, mrna_fasta_filename, "fasta")
        SeqIO.write(miRNA_record, mirna_fasta_filename, "fasta")

        miranda_cmd = "miranda {mir} {mrna} -out {out} -en 10000 -sc 60 ".format(mir=mirna_fasta_filename, mrna=mrna_fasta_filename, out=miranda_out)
        os.system (miranda_cmd)
        self.miranda_out_file = miranda_out

    def parse_miranda_output (self, miranda_file):

        def extract_seq (s):
            return s.split("'")[1].strip()[0:-2]

        def lines_that_contain(string, fp):
            return [line for line in fp if string in line]

        output=""

        fle = open (miranda_file,"r")
        mr = fle.readlines()
        fle.close()

        query_list = lines_that_contain("Query:", mr)
        info_list = lines_that_contain(">", mr)
        interaction_list = lines_that_contain("|", mr)
        ref_list = lines_that_contain("Ref:", mr)

        if len(query_list)<1:
            self.num_of_pairs = -1  # No hits
            raise NoMirandaHits

        query = extract_seq(query_list[0].strip("\n"))
        ref = extract_seq(ref_list[0].strip("\n"))
        interaction = interaction_list[0][query_list[0].find(query):query_list[0].find(query)+len(query)]
        c_info = info_list[2].split()
        site_loc = (int(c_info[6]), int(c_info[7]))

        mrna_bulge = ""
        mrna_inter = ""
        mir_inter = ""
        mir_bulge = ""

        query = query[::-1].upper()
        ref = ref[::-1].upper()
        interaction = interaction[::-1]
        self.num_of_pairs = 0

        for i in range (len(interaction)) :
            if interaction[i] == " " :
                mrna_inter += " "
                mir_inter += " "
                mrna_bulge += ref[i]
                mir_bulge += query[i]
            else :
                self.num_of_pairs+=1
                mrna_bulge += " "
                mrna_inter += ref[i]
                mir_inter += query[i]
                mir_bulge += " "
        mrna_bulge = mrna_bulge.replace ("-", " ")
        mrna_inter = mrna_inter.replace ("-", " ")
        mir_inter = mir_inter.replace ("-", " ")
        mir_bulge = mir_bulge.replace ("-", " ")

        self.miranda_presentation = ""
        self.miranda_presentation = self.miranda_presentation + "miranda" +"\n"
        self.miranda_presentation = self.miranda_presentation + "--------"  +"\n"
        self.miranda_presentation = self.miranda_presentation + ref +"\n"
        self.miranda_presentation = self.miranda_presentation + interaction +"\n"
        self.miranda_presentation = self.miranda_presentation + query +"\n"
        # mir_for_vienna = query.replace ("-", "")
        # mrna_for_vienna = ref.replace ("-", "")[::-1]
        # print "Vienna on miranda result"
        # print "-------------------------"
        # print ViennaRNADuplex (mir_for_vienna, mrna_for_vienna)

        self.mrna_coor = site_loc
        return InteractionRichPresentation (mrna_bulge, mrna_inter, mir_inter, mir_bulge)

    def __str__(self):
        return self.miranda_presentation + "\n" + str(self.IRP)



class NoMirandaHits(Exception):
    pass

