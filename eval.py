import numpy as np
from sklearn import metrics









def eval(ground_truths, prediction_scores, dir_to_print):
    precisions, recalls, pr_thresholds = metrics.precision_recall_curve(ground_truths, prediction_scores)
    precision_recall_area = metrics.auc(recalls, precisions)

    fpr, tpr, roc_thresholds = metrics.roc_curve(ground_truths, prediction_scores, pos_label=1)
    roc_area = metrics.auc(fpr, tpr)

    max_accuracy = find_max_accuracy(ground_truths, prediction_scores, pr_thresholds)



def find_max_accuracy(ground_truths, prediction_scores, pr_thresholds):


