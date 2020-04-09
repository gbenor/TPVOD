import os

script = "../Code/Classifier/ClassifierWithGridSearch.py"
flags="self-fit --feature_mode=without_hot_encoding"
yaml = "../Code/Classifier/yaml/classifiers_params_small.yaml"

for i in range (0,10):
# for i in range(10, 20):
    cmd = f"nice -n{i} python {script} {flags} {yaml} {i} {i+1}"
    print (cmd)
    # os.system(f"{cmd} &")

