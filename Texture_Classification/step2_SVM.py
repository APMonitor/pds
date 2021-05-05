from pathlib import Path

from PIL import Image                                   # for image I/O
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
        # print(cat_label)
        for img_path in cat_dir.glob('*.png'):
            img = Image.open(img_path.as_posix())
            arr = np.array(img)
            feature = compute_lbp(arr)
            vec.append(feature)
            cat.append(cat_label)
    return vec, cat

vec_train = np.load("vec_train.npy")
cat_train = np.load("cat_train.npy")
le = preprocessing.LabelEncoder()
le.fit(cat_train)
label_train = le.transform(cat_train)

vec_test, cat_test = load_data('test-set')              # load test data
label_test = le.transform(cat_test)

def get_conf_mat(y_pred, y_target, n_cats):
    """Build confusion matrix from scratch.
    (This part could be a good student assignment.)
    """
    conf_mat = np.zeros((n_cats, n_cats))
    n_samples = y_target.shape[0]
    for i in range(n_samples):
        _t = y_target[i]
        _p = y_pred[i]
        conf_mat[_t, _p] += 1
    norm = np.sum(conf_mat, axis=1, keepdims=True)
    return conf_mat / norm


def vis_conf_mat(conf_mat, cat_names, acc):
    """Visualize the confusion matrix and save the figure to disk."""
    n_cats = conf_mat.shape[0]

    fig, ax = plt.subplots()
    # figsize=(10, 10)

    cmap = cm.Blues
    im = ax.matshow(conf_mat, cmap=cmap)
    im.set_clim(0, 1)
    ax.set_xlim(-0.5, n_cats - 0.5)
    ax.set_ylim(-0.5, n_cats - 0.5)
    ax.set_xticks(np.arange(n_cats))
    ax.set_yticks(np.arange(n_cats))
    ax.set_xticklabels(cat_names)
    ax.set_yticklabels(cat_names)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    for i in range(n_cats):
        for j in range(n_cats):
            text = ax.text(j, i, round(
                conf_mat[i, j], 2), ha="center", va="center", color="w")

    cbar = fig.colorbar(im)

    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    _title = 'Normalized confusion matrix, acc={0:.2f}'.format(acc)
    ax.set_title(_title)

    # plt.show()
    _filename = 'conf_mat.png'
    plt.savefig(_filename, bbox_inches='tight')


# SVM
clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(vec_train, label_train)             # fit SVM using training data

# evaluation
prediction = clf.predict(vec_test)          # make prediction on the test data
# visualization
cmat = get_conf_mat(y_pred=prediction, y_target=label_test,
n_cats=len(le.classes_))
acc = cmat.trace() / cmat.shape[0]
vis_conf_mat(cmat, le.classes_, acc)
