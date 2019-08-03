#%%
from __future__ import division, print_function, unicode_literals

import numpy as np
import os

np.random.seed(42)

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

#matplotlib.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT_DIR = "/"
CHAPTER_ID = "classification"

def save_fig(fig_id, tight_layout=True) :
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id+".png")
    if tight_layout :
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

#%%
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]
y = y.astype(np.int)

#%%
some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")
save_fig("some_digit_plot")
plt.show()
