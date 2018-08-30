from generate_spoa_pileups_from_bam import sequence_to_index
from handlers.FileManager import FileManager
from handlers.DataLoader import DataLoader
from models.Rnn import Decoder
from column_consensus import *
from matplotlib import pyplot
from torch import nn
from torch import optim
from os import path
import torch
import datetime


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
            pyplot.text(i, j, "%.3f"%confusion, va="center", ha="center")

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

    # print(confusion.shape)

    y_target_indexes = torch.argmax(y, dim=1)
    y_predict_indexes = torch.argmax(y_predict, dim=1)

    # print(y_target_indexes.shape)

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

        confusion[target_index, predict_index] += 1

    return confusion


def test(model, data_loader):
    model.eval()

    total_confusion = None

    for b, batch in enumerate(data_loader):
        paths, x, y = batch

        n, h, w = x.shape
        # x = x.view([n,1,h,w])

        # y_predict shape = (batch_size, seq_len, hidden_size*num_directions)
        y_predict = model.forward(x)

        confusion = batch_sequential_confusion(y_predict=y_predict, y=y)

        # plot_prediction(x,y,y_predict)

        if total_confusion is None:
            total_confusion = confusion
        else:
            total_confusion += confusion

    total_confusion = normalize_confusion_matrix(total_confusion)
    plot_confusion(total_confusion)


def test_consensus(consensus_caller, data_loader):
    total_confusion = None

    for b, batch in enumerate(data_loader):
        paths, x, y = batch

        # (n,h,w) shape
        batch_size, height, width = x.shape

        for n in range(batch_size):
            x_n = x[n,:,:].data.numpy()
            y_n = y[n,:,:]

            x_n = trim_empty_rows(x_n)

            y_predict_n = consensus_caller.call_consensus_as_one_hot(x_n)
            y_predict_n = torch.FloatTensor(y_predict_n)

            confusion = sequential_confusion(y_predict=y_predict_n, y=y_n)

            # normalized_frequencies = consensus_caller.call_consensus_as_normalized_frequencies(x_n)
            #
            # plot_consensus_prediction(x=x_n,y=y_n,y_predict=normalized_frequencies)

            if total_confusion is None:
                total_confusion = confusion
            else:
                total_confusion += confusion

    total_confusion = normalize_confusion_matrix(total_confusion)
    plot_confusion(total_confusion)


def run():
    model_state_path = "/home/ryan/code/nanopore_assembly/output/training_2018-8-30-14-31-58-3-242/model_checkpoint_4"
    directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_chr1_full/test"   # spoa 2 pass variants excluded

    file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=directory, file_extension=".npz", sort=False)

    # Architecture parameters
    hidden_size = 16
    input_channels = 1      # 1-dimensional signal
    output_size = 5         # '-','A','C','T','G' one hot vector
    n_layers = 3

    # Hyperparameters
    dropout_rate = 0

    # Training parameters
    batch_size_train = 1

    checkpoint_interval = 300

    data_loader = DataLoader(file_paths=file_paths, batch_size=batch_size_train)

    model = Decoder(hidden_size=hidden_size, input_size=input_channels, output_size=output_size, n_layers=n_layers, dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(model_state_path))

    consensus_caller = ConsensusCaller(sequence_to_index)

    test(model=model, data_loader=data_loader)
    test_consensus(consensus_caller=consensus_caller, data_loader=data_loader)


if __name__ == "__main__":
    run()
