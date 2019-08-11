from pathlib import Path
import os
debug = False
python_int = "/home/local/BGU-USERS/benorgi/anaconda3/bin/python3"

exclude = ["run", "Pipeline", "validation", "_add"]
p = Path("../Code/Datafile_prepare/")
files = list(p.glob('**/*.py'))

run_list = []
for p in files:
    flag = True
    for e in exclude:
        if p.match(f"*{e}*"):
            flag = False
    if flag:
        run_list.append(p)

run_list = [f"{python_int} {r.resolve()}" for r in run_list]
for r in run_list:
    #cmd = f"echo {r} | batch"
    #print (cmd)
    cmd = f"{r} {debug} &"
    print(r)
    os.system(cmd)
#os.system("atq")