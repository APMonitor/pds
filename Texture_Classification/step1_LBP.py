from pathlib import Path

from PIL import Image, ImageOps                         # for image I/O
import numpy as np                                      # N-D array module
import matplotlib.pyplot as plt                         # visualization module
# color map for confusion matrix
from matplotlib import cm

# open source implementation of LBP
from skimage.feature import local_binary_pattern
# data preprocessing module in scikit-learn
from sklearn import preprocessing
# SVM implementation in scikit-learn
from sklearn.svm import LinearSVC

plt.rcParams['font.size'] = 11

# LBP function params
radius = 3
n_points = 8 * radius
METHOD = 'uniform'
n_bins = n_points + 2

def compute_lbp(arr):
    """Find LBP of all pixels.
    Also perform Vectorization/Normalization to get feature vector.
    """
    lbp = local_binary_pattern(arr, n_points, radius, METHOD)
    lbp = lbp.ravel()
    # feature_len = int(lbp.max() + 1)
    feature = np.zeros(n_bins)
    for i in lbp:
        feature[int(i)] += 1
    feature /= np.linalg.norm(feature, ord=1)
    return feature

def load_data(tag='training-set'):
    """Load (training/test) data from the directory.
    Also do preprocessing to extra features. 
    """
    tag_dir = Path.cwd() / tag
    print(tag_dir)
    vec = []
    cat = []
    for cat_dir in tag_dir.iterdir():
        cat_label = cat_dir.stem
        for img_path in cat_dir.glob('*.png'):
            img = Image.open(img_path.as_posix())
            #print(img_path.as_posix(), img.mode)
            if img.mode != 'L':
                img = ImageOps.grayscale(img)
                img.save(img_path.as_posix())
            arr = np.array(img)
            feature = compute_lbp(arr)
            vec.append(feature)
            cat.append(cat_label)
    return vec, cat

vec_train, cat_train = load_data('training-set')        # load training data
np.save("vec_train.npy", vec_train)
np.save("cat_train.npy", cat_train)