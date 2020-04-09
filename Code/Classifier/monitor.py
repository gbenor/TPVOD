from pathlib import Path

import pandas as pd
df = pd.DataFrame()
for i in range(10, 20):
    results_dir = Path("Results") / f"self{i}"
    models = results_dir.glob("*.model")
    m = [m.stem for m in models]
    d = {"split" : [i] * len(m),
             "file" : m
        }
    df = df.append(pd.DataFrame(d))
vc = df["file"].value_counts()

# for index, value in vc.items():
    # if str(index).find("xgbs") > 0:
    #     print("Index : {}, Value : {}".format(index, value))
print (vc)
print (sum(vc))
print (len(vc))
