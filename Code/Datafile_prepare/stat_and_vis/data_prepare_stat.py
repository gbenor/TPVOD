from pathlib import Path
import json
import pandas as pd
data_prepare_log_dir = Path("Datafiles_Prepare/Logs/")
data_prepare_csv_dir = Path("Datafiles_Prepare/CSV/")

s = pd.DataFrame()
index = 0


paper_dic ={"Global mapping of miRNA-target" : "\cite{scheel2017global}",
            "Unambiguous Identification of miRNA:" : "\cite{grosswendt2014unambiguous}",
            "Pairing beyond the Seed Supports" : "\cite{broughton2016pairing}",
            "Mapping the Human miRNA" : "\cite{Helwak2014}",
            "miRNA–target chimeras reveal" : "\cite{darnell_moore2015mirna}"

            }

csv = list(data_prepare_csv_dir.glob('*duplex*.csv'))
# exclude = ["duplex"]
# csv_l = []
# for p in csv:
#     flag = True
#     for e in exclude:
#         if p.match(f"*{e}*"):
#             flag = False
#     if flag:
#         csv_l.append(p)
#

min_num_of_pairs=11

for f in data_prepare_log_dir.glob('*.json'):
    #print (f)
    json_data = f.open().read()
    data = json.loads(json_data)

    org = data["Organism"]
    paper_name = data["paper"]
    print (paper_name)
    paper = "None"
    for k,v in paper_dic.items():
        if paper_name.startswith(k) :
            paper = v
    pipe_in = data ["Pipeline input samples count"]
    valid_utr3 = data ["Pipeline valid blast results"]
    valid_mirna = data ["Pipeline valid miRNA_no_***"]
    csv_c = [c for c in csv if c.match(f"*{str(f.stem)[:20]}*")]
    csv_c = [c for c in csv_c if c.match(f"*{org}*")]

    print (csv_c)


    s.loc ["Organism", index] = org
    s.loc ["paper", index] = paper
    # s.loc ["Samples", index] = pipe_in
    # s.loc ["valid_utr3", index] = valid_utr3
    # s.loc ["valid_mirna", index] = valid_mirna
    if len(csv_c) == 1:
        dp_file = csv_c[0]
        df = pd.read_csv(dp_file)
        valid_duplex = sum(df["num_of_pairs"] >= min_num_of_pairs)
        valid_seeds = sum(df["valid_seed"])

        s.loc ["Valid duplex", index] = valid_duplex
        s.loc ["Valid seeds", index] = valid_seeds





    index+=1
lat = s.to_latex()
lat = lat.replace("\\textbackslash cite", "\\cite")
lat = lat.replace("\\{", "{")
lat = lat.replace("\\}", "}")
lat = lat.replace("l}", "l|}")
lat = lat.replace("{l", "{||l")
lat = lat.replace("ll", "l|l|")



print (lat)
