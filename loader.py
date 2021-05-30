"""
Provides helper functions to load individual scans and patients
"""

import os
import numpy as np
import pydicom as dicom
from scipy import ndimage

def load_scan(scan_dir, transform=None):
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
    if transform is not None:
        vol = transform(vol)
    vol = vol.astype(np.int16)
    return vol

def load_cases(df, data_dir, n_scans = None, transform=None):
    if n_scans is None:
        n_scans = df.shape[0]
    scans = []
    for filepath in df['filepath'].tolist()[:n_scans]:
        print(f"Loading patient {filepath.split('/')[0]}!")
        fp = os.path.join(data_dir, filepath)
        img = load_scan(fp, transform)
        scans.append(img)
    print(f"Loaded {len(scans)} scans!")
    return scans

def resize_volume(img, desired_width=128, desired_height=128, desired_depth=32):
    """Resize across z-axis"""
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    # img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

# def main():
#     # data_dir = '/Users/tomasbencomo/code/cs235-data'
#     # patient = 'Breast_MRI_001'
#     # uid = '1.3.6.1.4.1.14519.5.2.1.186051521067863971269584893740842397538'
#     # date = '01-01-1990'
#     # scan_dir = os.path.join(data_dir, patient, uid, date)
#     # dirs = [f for f in os.listdir(scan_dir) if '.json' not in f]
#     # print(f"There are {len(dirs)} scan dirs")
#     img = load_scan('/Users/tomasbencomo/code/cs235-data/Breast_MRI_001/1.3.6.1.4.1.14519.5.2.1.186051521067863971269584893740842397538/01-01-1990/1.3.6.1.4.1.14519.5.2.1.175414966301645518238419021688341658582')
#     for i in range(img.shape[2]):
#         plt.imshow(img[:, :, i])
#         plt.show()
#     # scans = []
#     # for d in dirs:
#     #     img = load_scan(os.path.join(scan_dir, d))
#     #     scans.append(img)
#     #     print(d)
#     #     print(img.shape)

# if __name__ == '__main__':
#     main()

