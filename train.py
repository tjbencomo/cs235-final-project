import loader
import numpy as np
import pandas as pd

def load_cases(df, data_dir, n_scans = None):
    if n_scans is None:
        n_scans = df.shape[0]
    scans = []
    for filepath in df['filepath'].tolist()[:n_scans]:
        fp = os.path.join(data_dir, filepath)
        img = loader.load_scan(fp)
        scans.append(img)
    print(f"Loaded {len(scans)} scans!")
    return scans

def main():
    metadatafp = 'image_metadata.csv'
    df = pd.read_csv(metadatafp)
    data_dir = ''
    scans = load_cases(df, data_dir, 10)
    er_labels = df['ER']
    pr_labels = df['PR']
    her2_labels = df['HER2']

if __name__ == '__main__':
    main()
