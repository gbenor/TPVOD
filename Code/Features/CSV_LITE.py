import pandas as pd
import numpy as np
from Bio.Blast.Applications import NcbiblastnCommandline
from ViennaRNADuplex import *
from Bio import SeqIO
import random
from SeedFeatures import *
from SeedFeaturesCompact import *


from InteractionRichPresentation import *
from MirandaDuplex import *

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os
import itertools as it
from itertools import islice
from pathlib import Path
from Utils import *
import ExtractFeatures as ps
import JsonLog
import itertools
from multiprocessing import Pool
import multiprocessing


def main():
    input_dir = Path("Data/Features/CSV")
    output_dir = Path("Data/Features/CSV/LITE")
    col_to_drop = ['full_mrna', 'site_start','constraint', 'duplex_RNAplex_equals']

    for f in input_dir.glob('*.csv'):
        if f.is_dir():
            continue
        print (f)
        in_df = pd.read_csv(f)
        out_df = in_df.drop(col_to_drop, axis=1)
        for c in out_df.columns:
            if c.find("Unnamed") != -1:
                out_df.drop([c], axis=1, inplace=True)

        out_df.to_csv(Path(output_dir)/f.name)


if __name__ == "__main__":
    main()
