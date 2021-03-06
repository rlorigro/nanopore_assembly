from handlers.FileManager import FileManager
from modules.pileup_utils import trim_empty_rows
from handlers.DataLoader import DataLoader
from modules.ConsensusCaller import *
from modules.train_test_utils import realign_consensus_to_reference, calculate_accuracy_from_confusion
from matplotlib import pyplot
import torch


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

    axes.set_xticklabels(["","-","A","G","T","C"])
    axes.set_yticklabels(["","-","A","G","T","C"])

    for i in range(x):
        for j in range(y):
            confusion = confusion_matrix[i,j]
            pyplot.text(j, i, "%.0f" % confusion, va="center", ha="center")   # indexes are so confusing

    pyplot.imshow(confusion_matrix, cmap="viridis")
    pyplot.show()


# def trim_empty_rows(matrix):
#     h, w = matrix.shape
#
#     sums = numpy.sum(matrix, axis=1)
#     mask = numpy.nonzero(sums)
#
#     matrix = matrix[mask,:]
#     matrix = matrix.reshape([matrix.shape[1], matrix.shape[2]])
#
#     return matrix


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
        x = x.view([n,1,h,w])

        # y_predict shape = (batch_size, seq_len, hidden_size*num_directions)
        y_predict = model.forward(x)

        confusion = batch_sequential_confusion(y_predict=y_predict, y=y)

        if total_confusion is None:
            total_confusion = confusion
        else:
            total_confusion += confusion

    total_confusion = normalize_confusion_matrix(total_confusion)
    plot_confusion(total_confusion)


def test_consensus(consensus_caller, data_loader, n_batches=None):
    if n_batches is None:
        n_batches = len(data_loader)

    total_confusion = None
    total_realigned_confusion = None

    for b, batch in enumerate(data_loader):
        sys.stdout.write("\r %.2f%% COMPLETED  " % (100*b/n_batches))

        paths, x, y = batch

        # (n,h,w) shape
        batch_size, height, width = x.shape

        for n in range(batch_size):
            x_n = x[n,:,:].data.numpy()
            y_n = y[n,:,:].data.numpy()

            x_n = trim_empty_rows(x_n, background_value=sequence_to_float["-"])

            y_predict_n = consensus_caller.call_consensus_as_one_hot(x_n)

            consensus_sequence = consensus_caller.decode_one_hot_to_string(y_predict_n)
            reference_sequence = consensus_caller.decode_one_hot_to_string(y_n)

            # print(consensus_sequence)
            # print(reference_sequence)

            if consensus_sequence == '':
                pyplot.imshow(y_predict_n)
                pyplot.show()
                pyplot.close()
                pyplot.imshow(x_n)
                pyplot.show()
                pyplot.close()

            y_predict_n = torch.FloatTensor(y_predict_n)
            y_n = torch.FloatTensor(y_n)
            confusion = sequential_confusion(y_predict=y_predict_n, y=y_n)

            # realign strings to each other and convert to one hot
            y_pileup_predict_expanded, y_pileup_expanded = \
                realign_consensus_to_reference(consensus_sequence=consensus_sequence,
                                               ref_sequence=reference_sequence, print_alignment=False)

            y_pileup_predict_expanded = torch.FloatTensor(y_pileup_predict_expanded)
            y_pileup_expanded = torch.FloatTensor(y_pileup_expanded)
            realigned_confusion = sequential_confusion(y_predict=y_pileup_predict_expanded, y=y_pileup_expanded)

            # normalized_frequencies = consensus_caller.call_consensus_as_normalized_frequencies(x_n)
            # plot_consensus_prediction(x=x_n,y=y_n,y_predict=normalized_frequencies)

            if total_confusion is None:
                total_confusion = confusion
                total_realigned_confusion = realigned_confusion
            else:
                total_confusion += confusion
                total_realigned_confusion += realigned_confusion

        if b == n_batches:
            break

    print()

    plot_confusion(total_confusion)
    plot_confusion(total_realigned_confusion)

    # total_confusion = normalize_confusion_matrix(total_confusion)
    # total_realigned_confusion = normalize_confusion_matrix(total_realigned_confusion)
    #
    # plot_confusion(total_confusion)
    # plot_confusion(total_realigned_confusion)

    accuracy = calculate_accuracy_from_confusion(total_realigned_confusion)

    print("Total accuracy", accuracy)


def run():
    # model_state_path = "/home/ryan/code/nanopore_assembly/output/training_2018-8-28-17-13-26-1-240/model_checkpoint_15"
    # model_state_path = "/home/ryan/code/nanopore_assembly/output/training_2018-8-29-12-11-15-2-241/model_checkpoint_21"
    # model_state_path = "/home/ryan/code/nanopore_assembly/output/training_2018-8-30-11-49-32-3-242/model_checkpoint_43"
    # directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_chr1_full/test"   # spoa 2 pass variants excluded
    # directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-9-4-17-30-38-1-247"   # arbitrary 2500 window test region
    directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_celegans_chr1_1mbp_NONRUNLENGTH_2018-9-19"   # c elegans
    # directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_human_chr1_1mbp_NONRUNLENGTH_2018-9-18"      # human

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
    n_batches = 5000

    checkpoint_interval = 300

    data_loader = DataLoader(file_paths=file_paths, batch_size=batch_size_train)

    # model = EncoderDecoder(hidden_size=hidden_size, input_size=input_channels, output_size=output_size, n_layers=n_layers, dropout_rate=dropout_rate)
    # model.load_state_dict(torch.load(model_state_path))

    consensus_caller = ConsensusCaller(sequence_to_index, sequence_to_float)

    # test(model=model, data_loader=data_loader)

    print(n_batches, len(data_loader))
    test_consensus(consensus_caller=consensus_caller, data_loader=data_loader, n_batches=n_batches)


if __name__ == "__main__":
    run()
