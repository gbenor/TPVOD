import string
import random
from pathlib import Path
from datetime import datetime


def random_string (N):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))

def make_tmp_dir (path, parents=False):
    p = Path(path) / random_string(8)
    p.mkdir(mode=0o777, parents=parents, exist_ok=False)
    return p



def filename_suffix_append(f, s):
    f = Path(f)
    return str(f.parent / Path(f.stem + s + f.suffix))

def filename_date_append (f):
    return str(filename_suffix_append (f, "_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))))


def drop_unnamed_col(df):
    for c in df.columns:
        if c.find ("Unnamed")!=-1:
            df.drop([c], axis=1, inplace=True)
