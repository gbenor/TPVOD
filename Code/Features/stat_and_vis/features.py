from pathlib import Path
import json
import pandas as pd
import itertools
feature_csv_dir = Path("Features/CSV")
from collections import Counter



for f in feature_csv_dir.glob('*.csv'):
    if f.match("*neg*"):
        continue
    print (f)
    df = pd.read_csv(f)

    c = df.columns
    c_prefix = []
    flag = False
    for c in df.columns:
        if flag:
            try:
                if c.split("_")[0]!="Acc" and c.split("_")[0]!="miRNAMatchPosition":
                    c_prefix.append(c.split("_")[0]+c.split("_")[1])
                else:
                    c_prefix.append(c.split("_")[0])
            except IndexError:
                c_prefix.append(c.split("_")[0])
        if c=="constraint":
            flag=True


    #c_prefix = [c.split("_")[0]+c.split("_")[1] for c in df.columns]
    cnt = Counter (c_prefix)
    print (cnt)
    red_cnt = dict()
    for k in cnt:
        if cnt[k] > 0:
            red_cnt[k] = cnt[k]
    print (red_cnt)
    sum = 0
    for k in red_cnt:
        sum+=red_cnt[k]
    print (sum)
    feature_list = pd.DataFrame([red_cnt])

    ff = feature_list.transpose(copy=True)
    print (ff)
    ff.to_csv (feature_csv_dir / "feature_summary2.csv")
    print ("*******************************************************************")

    print (ff.to_latex(header=True, bold_rows=True))
   # send_mail("hello world", "hello gilad")

    break
print ("*******************************************************************")
