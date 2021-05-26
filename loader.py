import os
import numpy as np
import pydicom as dicom
import matplotlib.pyplot as plt

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

def main():
    data_dir = '/Users/tomasbencomo/code/cs235-data'
    patient = 'Breast_MRI_001'
    uid = '1.3.6.1.4.1.14519.5.2.1.186051521067863971269584893740842397538'
    date = '01-01-1990'
    scan_dir = os.path.join(data_dir, patient, uid, date)
    dirs = [f for f in os.listdir(scan_dir) if '.json' not in f]
    print(f"There are {len(dirs)} scan dirs")

    scans = []
    for d in dirs:
        img = load_scan(os.path.join(scan_dir, d))
        scans.append(img)
        print(d)
        print(img.shape)

if __name__ == '__main__':
    main()

