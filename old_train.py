import os
import loader
import cv2
from scipy import ndimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_cases(df, data_dir, n_scans = None):
    if n_scans is None:
        n_scans = df.shape[0]
    scans = []
    for filepath in df['filepath'].tolist()[:n_scans]:
        print(f"Loading patient {filepath.split('/')[0]}!")
        fp = os.path.join(data_dir, filepath)
        img = loader.load_scan(fp)
        scans.append(img)
    print(f"Loaded {len(scans)} scans!")
    return scans

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 256
    desired_height = 256
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

def main():
    metadatafp = 'image_metadata.csv'
    df = pd.read_csv(metadatafp)
    df = df[df['patient'] == 'Breast_MRI_001']
    data_dir = '../cs235-data/'
    # data_dir = '/home/tomasbencomo/final-project/data'
    scans = load_cases(df, data_dir, 1)
    er_labels = df['ER']
    pr_labels = df['PR']
    her2_labels = df['HER2']
    print("Completed loading!")
    for img in scans:
        print(img.shape)
        for i in range(img.shape[2]):
            plt.imshow(img[:, :, i])
            plt.show()
    reshaped = resize_volume(img)
    print(reshaped.shape)
    for i in range(reshaped.shape[2]):
        plt.imshow(reshaped[:, :, i])
        plt.show()


if __name__ == '__main__':
    main()
