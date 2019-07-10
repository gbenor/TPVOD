import pandas as pd
from Bio import SeqIO


mirbase_file = "MirBase/Release 22.1/mature.fa"

def read_mirbase_file(organism_prefix=None):
    fasta_sequences = SeqIO.parse(open(mirbase_file), 'fasta')
    miRBase_seq_list = []
    miRBase_dic = {}
    for fasta in fasta_sequences:
        mi_name, ma_name, mi_seq = fasta.id, fasta.description.split()[1], str(fasta.seq)
        if organism_prefix!=None:
            if not (mi_name.startswith(organism_prefix)):
                continue
        miRBase_dic[ma_name] = [mi_name, mi_seq]
        miRBase_seq_list.append(mi_seq)
    print ('number of entrences in miRBase: ', len(miRBase_dic))
    return miRBase_dic


def insert_mirna(in_df, col_name, organism_prefix=None):
    def find_mrna(row):
        mrna_id = row[col_name]
        list_of_mrna_seq = [l for k, l in miRBase_dic.items() if l[0] == mrna_id]
        try:
            return list_of_mrna_seq[0][1]
        except IndexError:
            return "No mrna match!!!"

    miRBase_dic = read_mirbase_file(organism_prefix)
    in_df['miRNA_seq'] = in_df.apply(find_mrna, axis=1)
    return in_df.copy()

def find_mirnas_by_prefix(miRBase_dic, prefix):
    prefix+="-"
    list_of_mrna_seq = [l for k, l in miRBase_dic.items() if l[0].startswith(prefix)]
    return list_of_mrna_seq

