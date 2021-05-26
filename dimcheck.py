import os
import numpy as np
import pydicom as dicom
import pandas as pd

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def load_scan(scan_dir):
    list_of_slices = os.listdir(scan_dir)
    locations = []
    slices = []
    for slicename in list_of_slices:
        if slicename.endswith('.dcm'):
            ds = dicom.read_file(os.path.join(scan_dir, slicename))
            locations.append(float(ds.SliceLocation))
            slices.append(ds.pixel_array)
    slices = [x for _,x in sorted(zip(locations, slices))]
    slices.reverse()
    vol = np.stack(slices, axis=-1)
    return vol

def search_directory(startdir):
    metadata = []
    patients = listdir_nohidden(startdir)
    for patient in patients:
        uid = list(listdir_nohidden(os.path.join(startdir, patient)))[0]
        date = list(listdir_nohidden(os.path.join(startdir, patient, uid)))[0]
        scan_dirs = listdir_nohidden(os.path.join(startdir, patient, uid, date))
        scan_dirs = [f for f in scan_dirs if '.json' not in f]
        for scd in scan_dirs:
            scan = load_scan(os.path.join(startdir, patient, uid, date, scd))
            metadata.append((patient, scan.shape[0], scan.shape[1], scan.shape[2]))
    return metadata

def main():
    sdir = '/Users/tomasbencomo/code/cs235-data'
    met = search_directory(sdir)
    df = pd.DataFrame(met, columns = ['patient', 'x', 'y', 'z'])
    df.to_csv('image_sizes.csv', index=False, header=True)

if __name__ == '__main__':
    main()

