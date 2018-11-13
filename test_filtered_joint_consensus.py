from modules.train_test_utils import *
from modules.pileup_utils import *
from handlers.FileManager import FileManager
from handlers.DataLoaderRunlength import DataLoader
from modules.ConsensusCaller import *
from modules.GapFilterer import GapFilterer
from matplotlib import pyplot
import torch


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


def sequential_confusion(y_predict, y, argmax=True):
    # x shape = (5, length)
    # y shape = (5, length)

    if y.shape[-1] != y_predict.shape[-1]:
        print("ERROR: incompatible target and prediction shape: ", y.shape, y_predict.shape)

    if argmax:
        n_classes, length = y_predict.shape

        y_target = torch.argmax(y, dim=0)
        y_predict = torch.argmax(y_predict, dim=0)

        confusion = numpy.zeros([n_classes, n_classes])

    else:
        y_predict = torch.LongTensor(y_predict)
        y = torch.LongTensor(y)

        length = y.shape[0]

        y_target = torch.argmax(y, dim=0).squeeze()

        confusion = numpy.zeros([5, 5])

        # print(y_target.shape)
        # print(y_predict.shape)

    mismatches = list()

    for l in range(length):
        target_index = y_target[l]
        predict_index = y_predict[l]

        if int(target_index) > 0 and int(predict_index) > 0 and target_index != predict_index:
            mismatches.append(l)

        # print(target_index)
        # print(predict_index)

        confusion[target_index, predict_index] += 1

    return confusion, mismatches


def sequential_confusion_from_index(y_predict, y, n_classes):
    # y shape = (1, length)

    print(y_predict.shape)
    print(y.shape)

    length = y_predict.shape[0]

    confusion = numpy.zeros([n_classes,n_classes])
    mismatches = list()

    for l in range(length):
        target_index = y[l]
        predict_index = y_predict[l]

        if int(target_index) > 0 and int(predict_index) > 0 and target_index != predict_index:
            mismatches.append(l)

        # print(target_index)
        # print(predict_index)

        confusion[target_index, predict_index] += 1

    return confusion, mismatches


def sequential_repeat_confusion(y_predict, y):
    # y shape = (length,)

    length = y_predict.shape[0]

    confusion = list()
    mismatches = list()

    for l in range(length):
        target_index = int(y[l])
        predict_index = int(y_predict[l])

        confusion.append((target_index, predict_index))

        if target_index != predict_index:
            mismatches.append(l)

    return confusion, mismatches


def test_consensus(consensus_caller, data_loader, n_batches, gap_filterer=None, plot_mismatches=False):
    total_sequence_confusion = None
    total_expanded_confusion = None
    total_repeat_confusion = list()

    for b, batch in enumerate(data_loader):
        sys.stdout.write("\r %.2f%% COMPLETED  " % (100*b/n_batches))

        paths, x_pileup, y_pileup_unfiltered, x_repeat, y_repeat_unfiltered, reversal = batch

        # print()
        # print("X PILEUP", x_pileup.shape)
        # print("Y PILEUP", y_pileup.shape)
        # print("X REPEAT", x_repeat.shape)
        # print("Y REPEAT", y_repeat.shape)
        # print("REVERSAL", reversal.shape)

        if gap_filterer is not None:

            batch = gap_filterer.filter_batch(batch)

            x_pileup, y_pileup, x_repeat, y_repeat, reversal = batch

        else:
            y_pileup = y_pileup_unfiltered
            y_repeat = y_repeat_unfiltered

        # x_pileup_n = x_pileup[0,:,:,:]
        # y_pileup_n = y_pileup[0,:,:,:]
        # x_repeat_n = x_repeat[0,:,:,:]
        # y_repeat_n = y_repeat[0,:,:,:]
        # reversal_n = reversal[0,:,:]

        # (n,h,w) shape
        batch_size, n_channels, height, width = x_pileup.shape
        batch_size, n_channels_unfiltered, height_unfiltered, width_unfiltered = y_pileup_unfiltered.shape

        for n in range(batch_size):
            # input shape = (batch_size, n_channels, height, width)
            # example x_pileup_n shape: (1, 5, 44, 24)
            # example y_pileup_n shape: (1, 5, 1, 24)
            # example x_repeat_n shape: (1, 1, 44, 24)
            # example y_repeat_n shape: (1, 1, 1, 24)

            x_pileup_n = x_pileup[n,:,:,:]
            y_pileup_n = y_pileup[n,:,:,:]
            y_pileup_unfiltered_n = y_pileup_unfiltered[n,:,:,:]
            x_repeat_n = x_repeat[n,:,:,:]
            y_repeat_n = y_repeat[n,:,:,:]
            y_repeat_unfiltered_n = y_repeat_unfiltered[n,:,:,:]

            # print()
            # print("X PILEUP", x_pileup_n.shape)
            # print("Y PILEUP", y_pileup_n.shape)
            # print("Y PILEUP", y_pileup_unfiltered_n.shape)
            # print("X REPEAT", x_repeat_n.shape)
            # print("Y REPEAT", y_repeat_n.shape)
            # print("Y REPEAT", y_repeat_unfiltered_n.shape)

            x_pileup_n = x_pileup_n.reshape([n_channels, height, width])
            y_pileup_n = y_pileup_n.reshape([n_channels, 1, width])
            y_pileup_unfiltered_n = y_pileup_unfiltered_n.reshape([n_channels, 1, width_unfiltered])

            # print()
            # print(x_pileup_n.shape)
            # print(y_pileup_n.shape)
            # print(x_repeat_n.shape)
            # print(y_repeat_n.shape)

            y_pileup_target = consensus_caller.call_consensus_as_index_from_one_hot(y_pileup_unfiltered_n)

            y_pileup_predict, y_repeat_predict = consensus_caller.call_joint_consensus_as_integers(pileup=x_pileup_n,
                                                                                                   repeat=x_repeat_n,
                                                                                                   reversal=reversal)

            print(y_repeat_predict.shape)

            x_repeat_n = x_repeat_n.reshape([height, width])
            y_repeat_n = y_repeat_n.reshape([width])
            y_repeat_unfiltered_n = y_repeat_unfiltered_n.reshape([width_unfiltered])

            if y_pileup_predict is None:
                print("ERROR: incorrect dimensions for pileup:")
                print(paths[0])
                continue

            # print()
            # print(y_pileup_predict.shape)
            # print(y_repeat_predict.shape)
            # print(y_pileup_target.shape)
            # print(y_repeat_n.shape)

            # decode as string to compare with non-runlength version
            expanded_consensus_string = \
                consensus_caller.expand_collapsed_consensus_as_string(consensus_indices=y_pileup_predict,
                                                                      repeat_consensus_integers=y_repeat_predict,
                                                                      ignore_spaces=True)

            # decode as string to compare with non-runlength version
            expanded_reference_string = \
                consensus_caller.expand_collapsed_consensus_as_string(consensus_indices=y_pileup_target,
                                                                      repeat_consensus_integers=y_repeat_unfiltered_n,
                                                                      ignore_spaces=True)

            # print(expanded_consensus_string)
            # print(expanded_reference_string)
            #

            print()
            y_pileup_predict_expanded, y_pileup_expanded, predict_string, target_string = \
                realign_consensus_to_reference(consensus_sequence=expanded_consensus_string,
                                               ref_sequence=expanded_reference_string,
                                               print_alignment=True,
                                               return_strings=True)

            # if numpy.any(y_repeat_predict != y_repeat_n):
            #     print(y_repeat_n)
            #     print(y_repeat_predict)
            #     # y_pileup_predict = y_pileup_predict.reshape([1, y_pileup_predict.shape[0]])
            #     y_repeat_predict_reshaped = y_repeat_predict.reshape([1, y_repeat_predict.shape[0]])
            #
            #     x_pileup_n_flat = flatten_one_hot_tensor(x_pileup_n)
            #     y_pileup_n_flat = flatten_one_hot_tensor(y_pileup_n)
            #     # y_pileup_predict_flat = flatten_one_hot_tensor(y_pileup_predict)
            #
            #     plot_runlength_prediction(x_pileup=x_pileup_n_flat, y_pileup=y_pileup_n_flat, x_repeat=x_repeat_n, y_repeat=y_repeat_predict_reshaped)

            y_pileup_predict_expanded = torch.FloatTensor(y_pileup_predict_expanded)
            y_pileup_expanded = torch.FloatTensor(y_pileup_expanded)

            expanded_confusion, _ = sequential_confusion(y_predict=y_pileup_predict_expanded, y=y_pileup_expanded)

            # y_pileup_predict = torch.FloatTensor(y_pileup_predict)
            # y_pileup_n = torch.FloatTensor(y_pileup_n)

            sequence_confusion, sequence_mismatch_indexes = sequential_confusion(y_predict=y_pileup_predict,
                                                                                 y=y_pileup_n,
                                                                                 argmax=False)

            repeat_confusion, repeat_mismatch_indexes = sequential_repeat_confusion(y_predict=y_repeat_predict, y=y_repeat_n)

            total_repeat_confusion.extend(repeat_confusion)

            if total_expanded_confusion is None:
                total_sequence_confusion = sequence_confusion
                total_expanded_confusion = expanded_confusion
            else:
                total_sequence_confusion += sequence_confusion
                total_expanded_confusion += expanded_confusion


            # counts = Counter(predict_string) + Counter(target_string)
            # print(counts)
            #
            # if counts['-'] > 5:
            #     print(paths[0])
            #
            #     y_repeat_n_plot = y_repeat_n.reshape([1, y_repeat_n.shape[0]])
            #     y_repeat_predict_n_plot = y_repeat_predict.reshape([1, y_repeat_predict.shape[0]])
            #     x_repeat_n_plot = x_repeat_n.reshape([height, width])
            #
            #     y_pileup_n_flat_plot = flatten_one_hot_tensor(y_pileup_n)
            #     y_pileup_predict_n_flat_plot = y_pileup_predict.reshape([1, y_pileup_predict.shape[0]])
            #     x_pileup_n_flat_plot = flatten_one_hot_tensor(x_pileup_n)
            #
            #     mismatch_index_strings = list(map(str,repeat_mismatch_indexes))
            #     print(mismatch_index_strings)
            #     title = "Repeat confusion at: " + ','.join(list(map(str,repeat_mismatch_indexes)))
            #
            #     plot_runlength_prediction_vs_truth(x_pileup=x_pileup_n_flat_plot,
            #                                        y_pileup_predict=y_pileup_predict_n_flat_plot,
            #                                        y_pileup=y_pileup_n_flat_plot,
            #                                        x_repeat=x_repeat_n_plot,
            #                                        y_repeat_predict=y_repeat_predict_n_plot,
            #                                        y_repeat=y_repeat_n_plot,
            #                                        title=title)


            # y_pileup_predict = y_pileup_predict.reshape([1, y_pileup_predict.shape[0]])
            # y_repeat_predict = y_repeat_predict.reshape([1, y_repeat_predict.shape[0]])
            #
            # x_pileup_n_flat = flatten_one_hot_tensor(x_pileup_n)
            # y_pileup_n_flat = flatten_one_hot_tensor(y_pileup_n)
            # y_pileup_predict_flat = flatten_one_hot_tensor(y_pileup_predict)

            # plot_runlength_prediction(x_pileup=x_pileup_n_flat, y_pileup=y_pileup_n_flat, x_repeat=x_repeat_n, y_repeat=y_repeat_predict)

        if b > 500:
            break

    print()

    # total_sequence_confusion = normalize_confusion_matrix(total_sequence_confusion)
    # total_expanded_confusion = normalize_confusion_matrix(total_expanded_confusion)

    accuracy = calculate_accuracy_from_confusion(total_expanded_confusion)

    print("Total accuracy", accuracy)

    plot_confusion(total_sequence_confusion)
    plot_confusion(total_expanded_confusion)

    runlength_confusion_matrix = plot_repeat_confusion(total_repeat_confusion)

    runlength_accuracy = calculate_accuracy_from_confusion(runlength_confusion_matrix)

    print("Total runlength accuracy", runlength_accuracy)


def run():
    # directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-2-10-43-22-1-275/NC_003282.8"  # one-hot with anchors and reversal matrix Chr4
    directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-15-13-10-33-0-288/NC_003279.8"  # one hot with anchors and reversal matrix chr1

    file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=directory, file_extension=".npz", sort=False)

    # Training parameters
    batch_size_train = 1
    checkpoint_interval = 300

    n_batches = 1000

    data_loader = DataLoader(file_paths=file_paths, batch_size=batch_size_train, parse_batches=False)

    consensus_caller = ConsensusCaller(sequence_to_index, sequence_to_float)

    gap_filterer = GapFilterer()

    print(len(data_loader))
    test_consensus(consensus_caller=consensus_caller, data_loader=data_loader, plot_mismatches=False, gap_filterer=gap_filterer, n_batches=n_batches)


if __name__ == "__main__":
    run()
