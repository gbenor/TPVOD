import pandas as pd
from Bio import SeqIO
from pathlib import Path

mirbase_dir = Path("MirBase")
mirbase_file = mirbase_dir / "mirbase_unified.csv"

def mirbase_db_concat(dir):
    sub_dir = [x for x in dir.iterdir() if x.is_dir()]
    mirbase_df = pd.DataFrame(columns=["mi_name", "mi_seq"])

    for p in sub_dir:
        mirbase_f = p / "mature.fa"
        print (mirbase_f)
        with open(mirbase_f) as fasta_file:
            for seq_record in SeqIO.parse(fasta_file, 'fasta'):  # (generator)
                mirbase_df = mirbase_df.append(
                    pd.Series([seq_record.id, seq_record.seq], index=mirbase_df.columns), ignore_index=True)

    mirbase_df.to_csv(mirbase_file)

def mirbasefile_preprocess():
    def get_prefix (x):
        return x.split("-")[0]
    df = pd.read_csv(mirbase_file)
    print(f"Before preprocess {df.shape}")
    df.drop_duplicates(subset=["mi_name"], keep='first', inplace=True)
    df["prefix"] = df["mi_name"].apply(get_prefix)
    print(f"After preprocess {df.shape}")
    df.to_csv(mirbase_file)




def read_mirbase_file(organism_prefix=None):
    df = pd.read_csv(mirbase_file)
    if organism_prefix is not None:
        df=df[df["prefix"]==organism_prefix]
    print ('number of entrences in miRBase: ', df.shape)
    return df




#
#
#
# def read_mirbase_file(organism_prefix=None):
#     fasta_sequences = SeqIO.parse(open(mirbase_file), 'fasta')
#     miRBase_seq_list = []
#     miRBase_dic = {}
#     for fasta in fasta_sequences:
#         mi_name, ma_name, mi_seq = fasta.id, fasta.description.split()[1], str(fasta.seq)
#         if organism_prefix!=None:
#             if not (mi_name.startswith(organism_prefix)):
#                 continue
#         miRBase_dic[ma_name] = [mi_name, mi_seq]
#         miRBase_seq_list.append(mi_seq)
#     print ('number of entrences in miRBase: ', len(miRBase_dic))
#     return miRBase_dic


def insert_mirna(in_df, col_name, organism_prefix=None):
    def find_mrna(row):
        mirna_id = row[col_name]
        try:
            return miRBase_df[miRBase_df["mi_name"]==mirna_id]["mi_seq"].values[0]
        except IndexError:
            return "No mrna match!!!"

    miRBase_df = read_mirbase_file(organism_prefix)
    in_df['miRNA_seq'] = in_df.apply(find_mrna, axis=1)
    return in_df.copy()

# def find_mrna_by_id(miRBase_dic, mir_id):
#     list_of_mrna_seq = [l for k, l in miRBase_dic.items() if l[0] == mir_id]
#     try:
#         return list_of_mrna_seq[0][1]
#     except IndexError:
#         return None
#
#
#
# def find_mirnas_by_prefix(miRBase_dic, prefix):
#     prefix+="-"
#     list_of_mrna_seq = [l for k, l in miRBase_dic.items() if l[0].startswith(prefix)]
#     return list_of_mrna_seq


def main ():
    #mirbase_db_concat(mirbase_dir)
    mirbasefile_preprocess()


if __name__ == "__main__":
    main()


