from modules.alignment_utils import *
from modules.pileup_utils import *
from matplotlib import pyplot
import numpy
import torch
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))


def realign_expanded_consensus_to_reference(expanded_sequence, ref_sequence, print_alignment=False):
    alignments, ref_alignment = get_spoa_alignment(sequences=[expanded_sequence], ref_sequence=ref_sequence, two_pass=False)
    ref_alignment = [ref_alignment]

    if print_alignment:
        print(alignments[0][1])
        print(ref_alignment[0][1])

    pileup_matrix = convert_aligned_reference_to_one_hot(alignments)
    reference_matrix = convert_aligned_reference_to_one_hot(ref_alignment)

    return pileup_matrix, reference_matrix


def normalize_confusion_matrix(confusion_matrix):
    sums = numpy.sum(confusion_matrix, axis=0)  # truth axis (as opposed to predict axis)

    confusion_matrix = confusion_matrix/sums

    return confusion_matrix


def plot_prediction(x, y, y_predict, save_path=None, title=""):
    if type(x) == torch.Tensor:
        x = x.data.numpy()

    if type(y) == torch.Tensor:
        y = y.data.numpy()

    if type(y_predict) == torch.Tensor:
        y_predict = y_predict.data.numpy()

    x_ratio = x.shape[0]/y_predict.shape[0]
    y_ratio = y.shape[0]/y_predict.shape[0]

    fig, axes = pyplot.subplots(nrows=3, gridspec_kw={'height_ratios': [1, y_ratio, x_ratio]})

    x_data = x[:,:].squeeze()
    y_target_data = y[:,:].squeeze()
    y_predict_data = y_predict[:,:].squeeze()

    axes[2].imshow(x_data)
    axes[1].imshow(y_target_data)
    axes[0].imshow(y_predict_data)

    axes[2].set_ylabel("x")
    axes[1].set_ylabel("y")
    axes[0].set_ylabel("y*")

    axes[0].set_title(title)

    if save_path is not None:
        pyplot.savefig(save_path)
    else:
        pyplot.show()
        pyplot.close()


def plot_consensus_prediction(x, y, y_predict, save_path=None, title=""):
    ratio = x.shape[0]/y.shape[0]
    fig, axes = pyplot.subplots(nrows=3, gridspec_kw={'height_ratios': [1, 1, ratio]})

    x_data = x.squeeze()
    y_target_data = y.squeeze()
    y_predict_data = y_predict.squeeze()

    axes[2].imshow(x_data)
    axes[1].imshow(y_target_data)
    axes[0].imshow(y_predict_data)

    axes[2].set_ylabel("x")
    axes[1].set_ylabel("y")
    axes[0].set_ylabel("y*")

    pyplot.title(title)

    if save_path is not None:
        pyplot.savefig(save_path)
    else:
        pyplot.show()
        pyplot.close()


def plot_confusion(confusion_matrix):
    x,y = confusion_matrix.shape

    axes = pyplot.axes()
    axes.set_ylabel("True class")
    axes.set_xlabel("Predicted class")

    axes.set_xticklabels(["","-","A","T","C","G"])
    axes.set_yticklabels(["","-","A","T","C","G"])

    for i in range(x):
        for j in range(y):
            confusion = confusion_matrix[i,j]
            pyplot.text(j, i, "%.3f"%confusion, va="center", ha="center")

    pyplot.imshow(confusion_matrix, cmap="viridis")
    pyplot.show()


def plot_repeat_confusion(total_repeat_confusion):
    axes = pyplot.axes()
    axes.set_ylabel("True run length")
    axes.set_xlabel("Predicted run length")

    x,y = zip(*total_repeat_confusion)

    n = max(x) + 1
    m = max(y) + 1

    confusion_matrix = numpy.zeros([n,m])

    for x,y in total_repeat_confusion:
        confusion_matrix[x,y] += 1

    confusion_matrix = normalize_confusion_matrix(confusion_matrix)

    pyplot.imshow(confusion_matrix, cmap="viridis")
    pyplot.show()
