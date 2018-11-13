from modules.ConsensusCaller import ConsensusCaller
from modules.alignment_utils import *
from modules.pileup_utils import *
from matplotlib import pyplot
import matplotlib
from os import path
import datetime
import numpy
import torch
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))


class ResultsHandler:
    def __init__(self):
        self.datetime_string = '-'.join(list(map(str, datetime.datetime.now().timetuple()))[:-1])
        self.subdirectory_name = "training_" + self.datetime_string

        self.output_directory_name = "output/"
        self.directory = path.join(self.output_directory_name, self.subdirectory_name)

        self.n_checkpoints = 0

        FileManager.ensure_directory_exists(self.directory)

    def save_plot(self, losses):
        loss_plot_filename = path.join(self.directory, "loss.png")

        figure = pyplot.figure()
        axes = pyplot.axes()
        axes.plot(losses)
        pyplot.savefig(loss_plot_filename)

    def save_model(self, model):
        self.n_checkpoints += 1

        model_filename = path.join(self.directory, "model_checkpoint_%d" % self.n_checkpoints)
        torch.save(model.state_dict(), model_filename)

        print("MODEL SAVED: ", model_filename)

    def save_config(self, model):
        pass


def calculate_accuracy_from_confusion(matrix, exclude_gaps=False):
    if exclude_gaps:
        matrix = matrix[1:,1:]

    n, m = matrix.shape
    diagonal_mask = numpy.eye(n,m, dtype=numpy.bool)
    off_diagonal_mask = numpy.invert(diagonal_mask)

    true_positives = numpy.sum(matrix[diagonal_mask])
    false_positives = numpy.sum(matrix[off_diagonal_mask])

    accuracy = true_positives / (true_positives + false_positives)

    return accuracy


def realign_consensus_to_reference(consensus_sequence, ref_sequence, print_alignment=False, return_strings=False):
    alignments, ref_alignment = get_spoa_alignment(sequences=[consensus_sequence], ref_sequence=ref_sequence, two_pass=False)
    # ref_alignment = [ref_alignment]

    if print_alignment:
        print(ref_alignment[0])
        print(alignments[0])

    pileup_matrix = convert_aligned_reference_to_one_hot(alignments[0])
    reference_matrix = convert_aligned_reference_to_one_hot(ref_alignment[0])

    if return_strings:
        return pileup_matrix, reference_matrix, alignments[0], ref_alignment[0]
    else:
        return pileup_matrix, reference_matrix


def normalize_confusion_matrix(confusion_matrix):
    sums = numpy.sum(confusion_matrix, axis=0)  # truth axis (as opposed to predict axis)

    confusion_matrix = confusion_matrix/sums

    return confusion_matrix


def label_pileup_encoding_plot(matrix, axis):
    consensus_caller = ConsensusCaller(sequence_to_index=sequence_to_index, sequence_to_float=sequence_to_float)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            index = numpy.round(matrix[i,j]).astype(numpy.uint8)

            # if value == 0:
            #     index = 0
            # else:
            #     index = consensus_caller.float_to_index[value]

            character = consensus_caller.index_to_sequence[index]

            axis.text(j,i,character, ha="center", va="center", fontsize=6)


def label_repeat_encoding_plot(matrix, axis):
    consensus_caller = ConsensusCaller(sequence_to_index=sequence_to_index, sequence_to_float=sequence_to_float)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i,j]

            axis.text(j,i,str(int(value)), ha="center", va="center", fontsize=6)


def plot_runlength_prediction_vs_truth(x_pileup, x_repeat, y_pileup, y_repeat, y_pileup_predict, y_repeat_predict, show_plot=True, save_path=None, title="", squeeze=False):
    if type(x_pileup) == torch.Tensor:
        x_pileup = x_pileup.data.numpy()

    if type(x_repeat) == torch.Tensor:
        x_repeat = x_repeat.data.numpy()

    if type(y_pileup) == torch.Tensor:
        y_pileup = y_pileup.data.numpy()

    if type(y_repeat) == torch.Tensor:
        y_repeat = y_repeat.data.numpy()

    if type(y_pileup_predict) == torch.Tensor:
        y_pileup_predict = y_pileup_predict.data.numpy()

    if type(y_repeat_predict) == torch.Tensor:
        y_repeat_predict = y_repeat_predict.data.numpy()

    # x_pileup = x_pileup[:, :]
    # x_repeat = x_repeat[:, :]

    if squeeze:
        x_pileup .squeeze()
        x_repeat .squeeze()

    print()
    print("x_pileup", x_pileup.shape)
    print("y_pileup", y_pileup.shape)
    print("y_pileup_predict", y_pileup_predict.shape)
    print("x_repeat", x_repeat.shape)
    print("y_repeat", y_repeat.shape)
    print("y_repeat_predict", y_repeat_predict.shape)

    y_pileup_ratio = y_pileup.shape[-2]
    y_pileup_predict_ratio = y_pileup_predict.shape[-2]/y_pileup.shape[-2]
    x_pileup_ratio = x_pileup.shape[-2]/y_pileup.shape[-2]
    x_repeat_ratio = x_repeat.shape[-2]/y_pileup.shape[-2]
    y_repeat_predict_ratio = y_repeat_predict.shape[-2]/y_pileup.shape[-2]
    y_repeat_ratio = y_repeat.shape[-2]/y_pileup.shape[-2]

    height_ratios = [y_pileup_ratio, y_pileup_predict_ratio, x_pileup_ratio, y_repeat_ratio, y_repeat_predict_ratio, x_repeat_ratio]

    print(height_ratios)

    fig, axes = pyplot.subplots(nrows=6, gridspec_kw={'height_ratios': height_ratios})

    label_pileup_encoding_plot(matrix=y_pileup, axis=axes[0])
    label_pileup_encoding_plot(matrix=y_pileup_predict, axis=axes[1])
    label_pileup_encoding_plot(matrix=x_pileup, axis=axes[2])
    label_repeat_encoding_plot(matrix=y_repeat, axis=axes[3])
    label_repeat_encoding_plot(matrix=y_repeat_predict, axis=axes[4])
    label_repeat_encoding_plot(matrix=x_repeat, axis=axes[5])

    red = numpy.array([255,19,0])/255
    yellow = numpy.array([255,230,10])/255
    blue = numpy.array([1,100,209])/255
    green = numpy.array([0,255,40])/255

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["black", blue, green, yellow, red])

    axes[0].imshow(y_pileup, cmap=cmap, vmin=0, vmax=4)
    axes[1].imshow(y_pileup_predict, cmap=cmap, vmin=0, vmax=4)
    axes[2].imshow(x_pileup, cmap=cmap, vmin=0, vmax=4)

    max_repeat = numpy.max(x_repeat)
    axes[3].imshow(y_repeat, vmin=0, vmax=max_repeat)
    axes[4].imshow(y_repeat_predict, vmin=0, vmax=max_repeat)
    axes[5].imshow(x_repeat, vmin=0, vmax=max_repeat)

    axes[0].set_ylabel("y")
    axes[1].set_ylabel("y*")
    axes[2].set_ylabel("x sequence")
    axes[3].set_ylabel("y")
    axes[4].set_ylabel("y*")
    axes[5].set_ylabel("x repeats")

    axes[0].tick_params(
        bottom=False,  # ticks along the bottom edge are off
        left=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off

    axes[1].tick_params(
        bottom=False,  # ticks along the bottom edge are off
        left=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off

    axes[3].tick_params(
        bottom=False,  # ticks along the bottom edge are off
        left=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off

    axes[4].tick_params(
        bottom=False,  # ticks along the bottom edge are off
        left=False,  # ticks along the top edge are off
        labelleft=False)  # labels along the bottom edge are off

    axes[0].set_title(title)

    if not save_path is None:
        pyplot.savefig(save_path)
    else:
        pyplot.show()

    pyplot.close()

    return axes, height_ratios


def plot_runlength_prediction(x_pileup, x_repeat, y_pileup, y_repeat, show_plot=True, save_path=None, title="", squeeze=False):
    if type(x_pileup) == torch.Tensor:
        x_pileup = x_pileup.data.numpy()

    if type(x_repeat) == torch.Tensor:
        x_repeat = x_repeat.data.numpy()

    if type(y_pileup) == torch.Tensor:
        y_pileup = y_pileup.data.numpy()

    if type(y_repeat) == torch.Tensor:
        y_repeat = y_repeat.data.numpy()

    # x_pileup = x_pileup[:, :]
    # x_repeat = x_repeat[:, :]

    if squeeze:
        x_pileup .squeeze()
        x_repeat .squeeze()

    print()
    print(x_pileup.shape)
    print(y_pileup.shape)
    print(x_repeat.shape)
    print(y_repeat.shape)

    x_pileup_ratio = x_pileup.shape[-2]/y_pileup.shape[-2]
    y_pileup_ratio = 1
    x_repeat_ratio = x_repeat.shape[-2]/y_pileup.shape[-2]
    y_repeat_ratio = y_repeat.shape[-2]/y_pileup.shape[-2]

    height_ratios = [y_pileup_ratio, x_pileup_ratio, y_repeat_ratio, x_repeat_ratio]

    fig, axes = pyplot.subplots(nrows=4, gridspec_kw={'height_ratios': height_ratios})

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

    if not save_path is None:
        pyplot.savefig(save_path)
    else:
        pyplot.show()

    pyplot.close()

    return axes, height_ratios


def plot_prediction(x, y, y_predict, save_path=None, title="", squeeze=False):
    if type(x) == torch.Tensor:
        x = x.data.numpy()

    if type(y) == torch.Tensor:
        y = y.data.numpy()

    if type(y_predict) == torch.Tensor:
        y_predict = y_predict.data.numpy()

    if squeeze:
        x_data = x[:,:].squeeze()
        y_target_data = y[:,:].squeeze()
        y_predict_data = y_predict[:,:].squeeze()
    else:
        x_data = x
        y_target_data = y
        y_predict_data = y_predict

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


def plot_binary_confusion(confusion_matrix):
    x,y = confusion_matrix.shape

    axes = pyplot.axes()
    axes.set_ylabel("True class")
    axes.set_xlabel("Predicted class")

    axes.set_xticks([0,1])
    axes.set_yticks([0,1])

    axes.set_xticklabels(["Gap","Not Gap"])
    axes.set_yticklabels(["Gap","Not Gap"])

    for i in range(x):
        for j in range(y):
            confusion = confusion_matrix[i,j]
            pyplot.text(j, i, "%.0f"%confusion, va="center", ha="center")

    pyplot.imshow(confusion_matrix, cmap="viridis")
    pyplot.show()


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
            pyplot.text(j, i, "%.0f"%confusion, va="center", ha="center")

    pyplot.imshow(confusion_matrix, cmap="viridis")
    pyplot.show()

    return confusion_matrix


def plot_repeat_confusion(total_repeat_confusion):
    axes = pyplot.axes()
    axes.set_ylabel("True run length")
    axes.set_xlabel("Predicted run length")

    x,y = zip(*total_repeat_confusion)

    n = max(x) + 1
    m = max(y) + 1

    print(n, m)

    confusion_matrix = numpy.zeros([n,m])

    for x,y in total_repeat_confusion:
        x = abs(x)
        y = abs(y)
        confusion_matrix[x,y] += 1

    normalized_confusion_matrix = normalize_confusion_matrix(confusion_matrix)

    pyplot.imshow(normalized_confusion_matrix, cmap="viridis")
    pyplot.show()

    return confusion_matrix
