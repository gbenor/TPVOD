from pathlib import Path
import pandas as pd


def main ():
    cvs_dir = Path("Datafiles_Prepare/CSV/")
    for x in cvs_dir.iterdir():
        print ("*"*80)
        print (x)
        df = pd.read_csv(x)
        print (df.columns)

if __name__ == "__main__":
    main()


