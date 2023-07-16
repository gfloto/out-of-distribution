import os, sys
import json
import numpy as np
from sklearn.metrics import roc_curve, auc

# aucroc for in-dist vs out-of-dist
def metrics(score_track, train_dataset):
    train_scores = score_track[train_dataset]
    train_labels = np.ones_like(train_scores)
    score_track.pop(train_dataset)

    metrics = {}
    for dataset, ood_scores in score_track.items():
        ood_labels = np.zeros_like(ood_scores)
        scores = np.concatenate((train_scores, ood_scores), axis=0)
        labels = np.concatenate((train_labels, ood_labels), axis=0)
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        aucroc = auc(fpr, tpr)

        metrics[dataset] = aucroc

    return metrics
    

