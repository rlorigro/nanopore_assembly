from modules.ConsensusCaller import ConsensusCaller
from modules.alignment_utils import *
from modules.pileup_utils import *
from matplotlib import pyplot
import numpy
import torch
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))


def realign_consensus_to_reference(consensus_sequence, ref_sequence, print_alignment=False):
    alignments, ref_alignment = get_spoa_alignment(sequences=[consensus_sequence], ref_sequence=ref_sequence, two_pass=False)
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


def label_pileup_encoding_plot(matrix, axis):
    consensus_caller = ConsensusCaller(sequence_to_index=sequence_to_index, sequence_to_float=sequence_to_float)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i,j]

            index = consensus_caller.float_to_index[value]
            character = consensus_caller.index_to_sequence[index]

            axis.text(j,i,character, ha="center", va="center", fontsize=6)


def label_repeat_encoding_plot(matrix, axis):
    consensus_caller = ConsensusCaller(sequence_to_index=sequence_to_index, sequence_to_float=sequence_to_float)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i,j]

            axis.text(j,i,str(int(value)), ha="center", va="center", fontsize=6)


def plot_runlength_prediction(x_pileup, x_repeat, y_pileup, y_repeat, save_path=None, title=""):
    if type(x_pileup) == torch.Tensor:
        x_pileup = x_pileup.data.numpy()

    if type(x_repeat) == torch.Tensor:
        x_repeat = x_repeat.data.numpy()

    if type(y_pileup) == torch.Tensor:
        y_pileup = y_pileup.data.numpy()

    if type(y_repeat) == torch.Tensor:
        y_repeat = y_repeat.data.numpy()

    x_pileup = x_pileup[:,:].squeeze()
    x_repeat = x_repeat[:,:].squeeze()
    # y_pileup = y_pileup[:,:].squeeze()
    # y_repeat = y_repeat[:,:].squeeze()

    print(x_pileup.shape)
    print(y_pileup.shape)
    print(x_repeat.shape)

    x_pileup_ratio = x_pileup.shape[-2]/y_pileup.shape[-2]
    y_pileup_ratio = 1
    x_repeat_ratio = x_repeat.shape[-2]/y_pileup.shape[-2]
    y_repeat_ratio = y_repeat.shape[-2]/y_pileup.shape[-2]

    fig, axes = pyplot.subplots(nrows=4, gridspec_kw={'height_ratios': [y_pileup_ratio,
                                                                        x_pileup_ratio,
                                                                        y_repeat_ratio,
                                                                        x_repeat_ratio]})

    label_pileup_encoding_plot(matrix=y_pileup, axis=axes[0])
    label_pileup_encoding_plot(matrix=x_pileup, axis=axes[1])
    label_repeat_encoding_plot(matrix=y_repeat, axis=axes[2])
    label_repeat_encoding_plot(matrix=x_repeat, axis=axes[3])


    axes[0].imshow(y_pileup)
    axes[1].imshow(x_pileup)
    axes[2].imshow(y_repeat)
    axes[3].imshow(x_repeat)

    axes[0].set_ylabel("y*")
    axes[1].set_ylabel("x sequence")
    axes[2].set_ylabel("y*")
    axes[3].set_ylabel("x repeats")

    axes[0].set_title(title)

    if save_path is not None:
        pyplot.savefig(save_path)
    else:
        pyplot.show()
        pyplot.close()


def plot_prediction(x, y, y_predict, save_path=None, title=""):
    if type(x) == torch.Tensor:
        x = x.data.numpy()

    if type(y) == torch.Tensor:
        y = y.data.numpy()

    if type(y_predict) == torch.Tensor:
        y_predict = y_predict.data.numpy()

    x_data = x[:,:].squeeze()
    y_target_data = y[:,:].squeeze()
    y_predict_data = y_predict[:,:].squeeze()

    x_ratio = x.shape[-2]/y_predict.shape[-2]
    y_ratio = y.shape[-2]/y_predict.shape[-2]

    fig, axes = pyplot.subplots(nrows=3, gridspec_kw={'height_ratios': [1, y_ratio, x_ratio]})

    axes[2].imshow(x_data)
    axes[1].imshow(y_target_data)
    axes[0].imshow(y_predict_data)

    axes[2].set_ylabel("x")
    axes[1].set_ylabel("y")
    axes[0].set_ylabel("y*")

    axes[0].set_title(title)

    if save_path is not None:
        pyplot.savefig(save_path)
        pyplot.close()
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

    axes[0].set_title(title)

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

    axes.set_xticklabels(["","-","A","G","T","C"])
    axes.set_yticklabels(["","-","A","G","T","C"])

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
