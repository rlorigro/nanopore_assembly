from generate_spoa_pileups_from_bam import sequence_to_index
from handlers.FileManager import FileManager
from handlers.DataLoaderRunlength import DataLoader
from generate_spoa_pileups_from_bam import get_spoa_alignment
from models.Cnn import EncoderDecoder
from column_consensus import *
from matplotlib import pyplot
from torch import nn
from torch import optim
from os import path
import torch
import datetime


def realign_expanded_consensus_to_reference(expanded_sequence, ref_sequence):
    alignments, ref_alignment = get_spoa_alignment(sequences=[expanded_sequence], ref_sequence=ref_sequence, two_pass=False)
    ref_alignment = [ref_alignment]

    # print(alignments[0][1])
    # print(ref_alignment[0][1])

    pileup_matrix = convert_aligned_reference_to_one_hot(alignments)
    reference_matrix = convert_aligned_reference_to_one_hot(ref_alignment)

    return pileup_matrix, reference_matrix


def normalize_confusion_matrix(confusion_matrix):
    sums = numpy.sum(confusion_matrix, axis=0)  # truth axis (as opposed to predict axis)

    confusion_matrix = confusion_matrix/sums

    return confusion_matrix


def plot_prediction(x, y, y_predict):
    fig, axes = pyplot.subplots(nrows=3, gridspec_kw={'height_ratios': [1, 1, 10]})

    x_data = x.data.numpy()[0, :, :, :].squeeze()
    y_target_data = y.data.numpy()[0, :, :].squeeze()
    y_predict_data = y_predict.data.numpy()[0, :, :].squeeze()

    # print(x_data.shape)

    axes[2].imshow(x_data)
    axes[1].imshow(y_target_data)
    axes[0].imshow(y_predict_data)

    axes[2].set_ylabel("x")
    axes[1].set_ylabel("y")
    axes[0].set_ylabel("y*")

    pyplot.show()
    pyplot.close()


def plot_consensus_prediction(x, y, y_predict):
    ratio = x.shape[0]/y.shape[0]
    fig, axes = pyplot.subplots(nrows=3, gridspec_kw={'height_ratios': [1, 1, ratio]})

    x_data = x.squeeze()
    y_target_data = y.squeeze()
    y_predict_data = y_predict.squeeze()

    # print(x_data.shape)

    axes[2].imshow(x_data)
    axes[1].imshow(y_target_data)
    axes[0].imshow(y_predict_data)

    axes[2].set_ylabel("x")
    axes[1].set_ylabel("y")
    axes[0].set_ylabel("y*")

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


def trim_empty_rows(matrix):
    h, w = matrix.shape

    sums = numpy.sum(matrix, axis=1)
    mask = numpy.nonzero(sums)

    matrix = matrix[mask,:].squeeze()

    return matrix


def sequential_loss_CE(y_predict, y, loss_fn):
    # x shape = (n, 5, length)
    # y shape = (n, 5, length)

    n, c, l = y_predict.shape

    y_target = torch.argmax(y, dim=1)

    loss = None

    for i in range(l):
        if i == 0:
            loss = loss_fn(y_predict[:,:,i], y_target[:,i])
        else:
            loss += loss_fn(y_predict[:,:,i], y_target[:,i])

    return loss


def batch_sequential_confusion(y_predict, y):
    # x shape = (n, 5, length)
    # y shape = (n, 5, length)

    batch_size, n_classes, length = y_predict.shape

    confusion = numpy.zeros([n_classes,n_classes])

    y_target_indexes = torch.argmax(y, dim=1)
    y_predict_indexes = torch.argmax(y_predict, dim=1)

    for l in range(length):
        for b in range(batch_size):
            # print(y_predict_indexes.shape, y_target_indexes.shape)
            # print(y_predict[b,:,l], y[b,:,l])

            target_index = y_target_indexes[b,l]
            predict_index = y_predict_indexes[b,l]

            # print(target_index, predict_index)

            confusion[target_index, predict_index] += 1

    return confusion


def sequential_confusion(y_predict, y):
    # x shape = (5, length)
    # y shape = (5, length)

    n_classes, length = y_predict.shape

    confusion = numpy.zeros([n_classes,n_classes])

    y_target = torch.argmax(y, dim=0)
    y_predict = torch.argmax(y_predict, dim=0)

    for l in range(length):
        target_index = y_target[l]
        predict_index = y_predict[l]

        # print(target_index)
        # print(predict_index)

        confusion[target_index, predict_index] += 1

    return confusion


def sequential_repeat_confusion(y_predict, y):
    # x shape = (5, length)
    # y shape = (5, length)

    n_classes, length = y_predict.shape

    # print(y_predict.shape)

    confusion = list()

    for l in range(length):
        target_index = int(y[:,l])
        predict_index = int(y_predict[:,l])

        confusion.append((target_index, predict_index))

    return confusion


def test_consensus(consensus_caller, data_loader):
    total_sequence_confusion = None
    total_expanded_confusion = None
    total_repeat_confusion = list()

    for b, batch in enumerate(data_loader):
        paths, x_pileup, y_pileup, x_repeat, y_repeat = batch

        # (n,h,w) shape
        batch_size, height, width = x_pileup.shape

        for n in range(batch_size):
            x_pileup_n = x_pileup[n,:,:].reshape([height,width])
            y_pileup_n = y_pileup[n,:,:].reshape([1,width])
            x_repeat_n = x_repeat[n,:,:].reshape([height,width])
            y_repeat_n = y_repeat[n,:,:].reshape([1,width])

            # remove padding
            x_pileup = trim_empty_rows(x_pileup_n)
            x_repeat = trim_empty_rows(x_repeat_n)

            # use consensus caller on bases and repeats independently
            y_pileup_predict = consensus_caller.call_consensus_as_encoding(x_pileup)
            y_repeat_predict = consensus_caller.call_repeat_consensus_as_integer_vector(repeat_matrix=x_repeat_n,
                                                                                        pileup_matrix=x_pileup_n,
                                                                                        consensus_encoding=y_pileup_predict)

            # decode as string to compare with non-runlength version
            expanded_consensus_string = \
                consensus_caller.expand_collapsed_consensus_as_string(consensus_encoding=y_pileup_predict,
                                                                      repeat_consensus_encoding=y_repeat_predict,
                                                                      ignore_spaces=True)

            # decode as string to compare with non-runlength version
            expanded_reference_string = \
                consensus_caller.expand_collapsed_consensus_as_string(consensus_encoding=y_pileup_n,
                                                                      repeat_consensus_encoding=y_repeat_n,
                                                                      ignore_spaces=True)

            # realign strings to each other and convert to one hot
            y_pileup_predict_expanded, y_pileup_expanded = \
                realign_expanded_consensus_to_reference(expanded_sequence=expanded_consensus_string,
                                                        ref_sequence=expanded_reference_string)

            y_pileup_predict_expanded = torch.FloatTensor(y_pileup_predict_expanded)
            y_pileup_expanded = torch.FloatTensor(y_pileup_expanded)

            expanded_confusion = sequential_confusion(y_predict=y_pileup_predict_expanded, y=y_pileup_expanded)

            # re-call consensus as one-hot vector series (to make confusion matrix easier)
            y_pileup_predict = consensus_caller.call_consensus_as_one_hot(y_pileup_predict)
            y_pileup_n = consensus_caller.call_consensus_as_one_hot(y_pileup_n)

            y_pileup_predict = torch.FloatTensor(y_pileup_predict)
            y_pileup_n = torch.FloatTensor(y_pileup_n)

            sequence_confusion = sequential_confusion(y_predict=y_pileup_predict, y=y_pileup_n)
            repeat_confusion = sequential_repeat_confusion(y_predict=y_repeat_predict, y=y_repeat_n)

            # normalized_frequencies = consensus_caller.call_consensus_as_normalized_frequencies(x_n)
            # plot_consensus_prediction(x=x_n,y=y_n,y_predict=normalized_frequencies)

            total_repeat_confusion.extend(repeat_confusion)

            if total_sequence_confusion is None:
                total_sequence_confusion = sequence_confusion
                total_expanded_confusion = expanded_confusion
            else:
                total_sequence_confusion += sequence_confusion
                total_expanded_confusion += expanded_confusion

    total_sequence_confusion = normalize_confusion_matrix(total_sequence_confusion)
    total_expanded_confusion = normalize_confusion_matrix(total_expanded_confusion)

    plot_confusion(total_sequence_confusion)
    plot_confusion(total_expanded_confusion)
    plot_repeat_confusion(total_repeat_confusion)


def run():
    directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-9-4-12-26-1-1-247"    # arbitrary test 800k

    file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=directory, file_extension=".npz", sort=False)

    # Training parameters
    batch_size_train = 1
    checkpoint_interval = 300

    data_loader = DataLoader(file_paths=file_paths, batch_size=batch_size_train, parse_batches=False)

    consensus_caller = ConsensusCaller(sequence_to_index, sequence_to_float)

    test_consensus(consensus_caller=consensus_caller, data_loader=data_loader)


if __name__ == "__main__":
    run()
