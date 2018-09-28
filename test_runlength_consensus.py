from modules.train_test_utils import *
from modules.pileup_utils import *
from handlers.FileManager import FileManager
from handlers.DataLoaderRunlength import DataLoader
from modules.ConsensusCaller import *
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


def sequential_confusion(y_predict, y):
    # x shape = (5, length)
    # y shape = (5, length)

    n_classes, length = y_predict.shape

    confusion = numpy.zeros([n_classes,n_classes])
    mismatches = list()

    y_target = torch.argmax(y, dim=0)
    y_predict = torch.argmax(y_predict, dim=0)

    for l in range(length):
        target_index = y_target[l]
        predict_index = y_predict[l]

        if int(target_index) > 0 and int(predict_index) > 0 and target_index != predict_index:
            mismatches.append(l)

        # print(target_index)
        # print(predict_index)

        confusion[target_index, predict_index] += 1

    return confusion, mismatches


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


def test_consensus(consensus_caller, data_loader, plot_mismatches=False):
    total_sequence_confusion = None
    total_expanded_confusion = None
    total_repeat_confusion = list()

    for b, batch in enumerate(data_loader):
        sys.stdout.write("\r %.2f%% COMPLETED  " % (100*b/len(data_loader)))

        paths, x_pileup, y_pileup, x_repeat, y_repeat = batch

        # (n,h,w) shape
        batch_size, n_channels, height, width = x_pileup.shape

        print()
        print(x_pileup.shape)
        print(y_pileup.shape)
        print(x_repeat.shape)
        print(y_repeat.shape)

        for n in range(batch_size):
            # print(y_pileup.shape)
            y_pileup_n = consensus_caller.encode_one_hot_as_float(y_pileup[n,:,:])

            x_pileup_n = x_pileup[n,:,:].reshape([n_channels,height,width])
            y_pileup_n = y_pileup_n.reshape([1,width])
            x_repeat_n = x_repeat[n,:,:].reshape([n_channels,height,width])
            y_repeat_n = y_repeat[n,:,:].reshape([1,width])

            # remove padding
            x_pileup_n = trim_empty_rows(x_pileup_n, background_value=sequence_to_float["-"])
            x_repeat_n = trim_empty_rows(x_repeat_n, background_value=sequence_to_float["-"])

            # use consensus caller on bases and repeats independently
            y_pileup_predict = consensus_caller.call_consensus_as_encoding(x_pileup_n)

            if y_pileup_predict is None:
                print("ERROR: incorrect dimensions for pileup:")
                print(paths[0])
                continue

            try:
                y_repeat_predict = \
                    consensus_caller.call_repeat_consensus_as_integer_vector(repeat_matrix=x_repeat_n,
                                                                             pileup_matrix=x_pileup_n,
                                                                             consensus_encoding=y_pileup_predict,
                                                                             use_model=True,
                                                                             use_prior=False)
            except IndexError as error:
                print(error)
                print(x_pileup_n.shape)
                print(x_repeat_n.shape)
                continue

            # decode as string to compare with non-runlength version
            expanded_consensus_string = \
                consensus_caller.expand_collapsed_consensus_as_string(consensus_encoding=y_pileup_predict,
                                                                      repeat_consensus_encoding=y_repeat_predict,
                                                                      ignore_spaces=True)

            # # decode as string to compare with non-runlength version
            expanded_reference_string = \
                consensus_caller.expand_collapsed_consensus_as_string(consensus_encoding=y_pileup_n,
                                                                      repeat_consensus_encoding=y_repeat_n,
                                                                      ignore_spaces=True)

            # realign strings to each other and convert to one hot
            y_pileup_predict_expanded, y_pileup_expanded = \
                realign_consensus_to_reference(consensus_sequence=expanded_consensus_string,
                                               ref_sequence=expanded_reference_string,
                                               print_alignment=False)

            y_pileup_predict_expanded = torch.FloatTensor(y_pileup_predict_expanded)
            y_pileup_expanded = torch.FloatTensor(y_pileup_expanded)
            expanded_confusion, _ = sequential_confusion(y_predict=y_pileup_predict_expanded, y=y_pileup_expanded)

            # re-call consensus as one-hot vector series (to make confusion matrix easier)
            y_pileup_predict = consensus_caller.call_consensus_as_one_hot(y_pileup_predict)
            y_pileup_n = consensus_caller.call_consensus_as_one_hot(y_pileup_n)

            y_pileup_predict = torch.FloatTensor(y_pileup_predict)
            y_pileup_n = torch.FloatTensor(y_pileup_n)

            sequence_confusion, mismatches = sequential_confusion(y_predict=y_pileup_predict, y=y_pileup_n)
            repeat_confusion = sequential_repeat_confusion(y_predict=y_repeat_predict, y=y_repeat_n)

            total_repeat_confusion.extend(repeat_confusion)

            if total_sequence_confusion is None:
                total_sequence_confusion = sequence_confusion
                total_expanded_confusion = expanded_confusion
            else:
                total_sequence_confusion += sequence_confusion
                total_expanded_confusion += expanded_confusion

            if plot_mismatches:
                if len(mismatches) > 0:
                    print(paths[0])
                    print(mismatches)
                    normalized_frequencies = consensus_caller.get_normalized_frequencies(x_pileup_n)

                    img_path = paths[0].split('/')[-1].split('.')[0]
                    title = str(mismatches)

                    print(img_path)
                    plot_consensus_prediction(x=x_pileup_n, y=y_pileup_n, y_predict=normalized_frequencies, save_path=img_path, title=title)

                    # plot_prediction(x=x_pileup_n, y=y_pileup_n.numpy(), y_predict=y_pileup_predict.numpy())

        if b > 10000:
            break

    # total_sequence_confusion = normalize_confusion_matrix(total_sequence_confusion)
    # total_expanded_confusion = normalize_confusion_matrix(total_expanded_confusion)

    accuracy = calculate_accuracy_from_confusion(total_expanded_confusion)

    print(accuracy)

    plot_confusion(total_sequence_confusion)
    plot_confusion(total_expanded_confusion)
    plot_repeat_confusion(total_repeat_confusion)


def run():
    # directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-9-4-12-26-1-1-247"    # arbitrary test 800k
    # directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-9-6-13-16-52-3-249"    # 2500 window test
    # directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_celegans_chr1_1mbp_2018-9-18"   # c elegans
    directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-9-28-14-53-16-4-271"      #one-hot encoding test

    file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=directory, file_extension=".npz", sort=False)

    # Training parameters
    batch_size_train = 1
    checkpoint_interval = 300

    data_loader = DataLoader(file_paths=file_paths, batch_size=batch_size_train, parse_batches=False)

    consensus_caller = ConsensusCaller(sequence_to_index, sequence_to_float)

    print(len(data_loader))
    test_consensus(consensus_caller=consensus_caller, data_loader=data_loader, plot_mismatches=False)


if __name__ == "__main__":
    run()
