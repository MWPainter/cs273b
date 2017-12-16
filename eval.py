import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics









def eval(ground_truths, prediction_scores, dir_to_print):
    precisions, recalls, pr_thresholds = metrics.precision_recall_curve(ground_truths, prediction_scores)
    precision_recall_area = metrics.auc(recalls, precisions)

    fpr, tpr, roc_thresholds = metrics.roc_curve(ground_truths, prediction_scores, pos_label=1)
    roc_area = metrics.auc(fpr, tpr)

    max_accuracy = find_max_accuracy(ground_truths, prediction_scores, pr_thresholds)

    printGraph(recalls, precisions, "precision vs recall", "recall", "precision", dir_to_print)
    printGraph(fpr, tpr, "ROC curve", "False Positive Rate", "True Positive Rate", dir_to_print)

    print "Max Accuracy: " + max_accuracy
    print "ROC AUC: " + roc_area
    print "PR AUC: " + precision_recall_area



def find_max_accuracy(ground_truths, prediction_scores, pr_thresholds):
    mx = 0.0
    for thresh in pr_thresholds:
        acc = metrics.accuracy_score(ground_truths, prediction_scores > thresh)
        if acc > mx: mx = acc
    return mx



def printGraph(xs, ys, graph_name, xlbl, ylbl, dir_to_print):
    sns.set_style("darkgrid")
    plt.plot(xs, ys, 'r', label = "Test")
    plt.legend(loc = "upper left")    
    plt.title(graph_name)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.savefig(dir_to_print + graph_name)
    plt.clf()
