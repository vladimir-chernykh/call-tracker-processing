import numpy as np

import matplotlib.pyplot as plt

from data_utils import Constants


def draw_confusion_matrix(confusion_matrix, params=Constants()):
    plt.pcolor(confusion_matrix.T, cmap=plt.cm.Reds)
    ax = plt.gca()
    ax.set_xticklabels(params.available_emotions, minor=False, fontsize=14)
    ax.set_yticklabels(params.available_emotions, minor=False, fontsize=14)
    ax.set_xticks(np.arange(len(params.available_emotions)) + 0.5, minor=False)
    ax.set_yticks(np.arange(len(params.available_emotions)) + 0.5, minor=False)
    ax.set_xlabel("Expert", fontsize=14)
    ax.set_ylabel("Model", fontsize=14)
    for i in range(len(params.available_emotions)):
        for j in range(len(params.available_emotions)):
            plt.text(i + 0.2, j + 0.4, str(round(confusion_matrix[i, j], 3)), fontsize=14)
