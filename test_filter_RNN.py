from modules.train_test_utils import *
from modules.pileup_utils import *
from handlers.FileManager import FileManager
from handlers.DataLoaderRunlength import DataLoader
from models.SimpleRnn import Decoder
from matplotlib import pyplot
import torch
from torch import optim
from torch import nn


def generate_roc_curve(y_predict, y_target):
    y_predict = y_predict.detach().numpy()
    y_target = y_target.detach().numpy()

    sensitivities = list()
    precisions = list()
    threshold_labels = list()

    range_size = 1000
    for i in range(range_size - 1):
        threshold = (float(1) / range_size) * (i + 1)

        print()
        print(threshold)

        predict_labels = numpy.copy(y_predict)

        zero_mask = (predict_labels < threshold)
        one_mask = (predict_labels > threshold)

        predict_labels[zero_mask] = 0
        predict_labels[one_mask] = 1

        # non_equivalency_mask = (predict_labels != truth_labels)
        true_positive_mask = (y_target == 1)
        true_negative_mask = (y_target == 0)

        n_true_positive = numpy.count_nonzero(predict_labels[true_positive_mask])
        n_false_positive = numpy.count_nonzero(predict_labels[true_negative_mask])
        n_true_negative = numpy.count_nonzero(true_negative_mask) - n_false_positive
        n_false_negative = numpy.count_nonzero(true_positive_mask) - n_true_positive

        print("TP: ", n_true_positive)
        print("FP: ", n_false_positive)
        print("TN: ", n_true_negative)
        print("FN: ", n_false_negative)

        if n_true_positive == 0:
            n_true_positive = 1e-15

        sensitivity = n_true_positive / (n_true_positive + n_false_negative)
        precision = n_true_positive / (n_true_positive + n_false_positive)

        print(sensitivity)
        print(precision)

        sensitivities.append(sensitivity)
        precisions.append(precision)

        if int(threshold * 1000) % int(0.1 * 1000) < 1:
            threshold_labels.append([sensitivity, precision, round(threshold, 1)])

        # print(zero_mask[5:10])
        # print(one_mask[5:10])
        # print(predict_labels[5:10])

    print("precision")
    print(precisions)
    print("sensitivity")
    print(sensitivities)

    ax = pyplot.axes()
    pyplot.gcf().set_size_inches(10, 8)
    ax.plot(sensitivities, precisions)
    ax.set_ylabel("Precision")
    ax.set_xlabel("Sensitivity")

    # ax.set_xlim(0,1.0)
    # ax.set_ylim(0,1.0)

    for label in threshold_labels:
        x, y, t = label
        ax.text(x, y, "T=" + str(t), ha="right", va="top")

    pyplot.show()


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


def sequential_binary_confusion(y_predict, y, threshold=0.5):
    # x shape = (5, length)
    # y shape = (5, length)

    n_classes, length = y_predict.shape

    confusion = numpy.zeros([2,2])
    mismatches = list()

    y_target = y.detach().numpy().reshape([length])
    y_predict = (y_predict.detach().numpy().reshape([length]) > threshold).astype(numpy.int8)

    for l in range(length):
        target_index = y_target[l]
        predict_index = y_predict[l]
        # print(target_index, predict_index)

        if int(target_index) > 0 and int(predict_index) > 0 and target_index != predict_index:
            mismatches.append(l)

        confusion[target_index, predict_index] += 1

    return confusion, mismatches


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
    # y shape = (length,)

    length = y_predict.shape[0]

    confusion = list()

    for l in range(length):
        target_index = int(y[l])
        predict_index = int(y_predict[l])

        confusion.append((target_index, predict_index))

    return confusion


def test(model, data_loader, n_batches, results_handler, checkpoint_interval, consensus_caller, use_gpu=False):
    model.eval()

    if use_gpu:
        print("USING GPU :)")
        x_dtype = torch.cuda.FloatTensor
        # y_dtype = torch.cuda.FloatTensor  # for MSE Loss or BCE loss
        y_dtype = torch.cuda.LongTensor  # for CE Loss

    else:
        x_dtype = torch.FloatTensor
        # y_dtype = torch.FloatTensor  # for MSE Loss or BCE loss
        y_dtype = torch.LongTensor  # for CE Loss

    total_sequence_confusion = None
    losses = list()

    all_predictions = None
    all_targets = None

    for b, batch in enumerate(data_loader):
        sys.stdout.write("\r %.2f%% COMPLETED  " % (100*b/n_batches))
        paths, x_pileup, y_pileup, x_repeat, y_repeat, reversal = batch

        # (n,h,w) shape
        batch_size, n_channels, height, width = x_pileup.shape
        reversal = reversal.reshape([1]+list(reversal.shape))

        # print()
        # print("X PILEUP",x_pileup.shape)
        # print("Y PILEUP",y_pileup.shape)
        # print("X REPEAT",x_repeat.shape)
        # print("Y REPEAT",y_repeat.shape)
        # print("REVERSAL",reversal.shape)

        x = torch.cat([x_pileup, x_repeat], dim=2).reshape([n_channels, 61, width])

        y_pileup_predict = model.forward(x)

        y_pileup_n = y_pileup[0,:,0,:]                  # ignore the depth dimension because y has always 1 "coverage"
        y_pileup_predict_n = y_pileup_predict[0,:,:]

        sequence_confusion, mismatches = sequential_binary_confusion(y_predict=y_pileup_predict_n, y=y_pileup_n)

        if total_sequence_confusion is None:
            total_sequence_confusion = sequence_confusion
        else:
            total_sequence_confusion += sequence_confusion

        if all_predictions is None:
            all_predictions = y_pileup_predict_n
            all_targets = y_pileup_n
        else:
            all_predictions = torch.cat([all_predictions, y_pileup_predict_n], dim=-1)
            all_targets = torch.cat([all_targets, y_pileup_n], dim=-1)

        if b == n_batches:
            break

    print()

    generate_roc_curve(y_predict=all_predictions, y_target=all_targets)

    # total_sequence_confusion = normalize_confusion_matrix(total_sequence_confusion)

    # accuracy = calculate_accuracy_from_confusion(total_expanded_confusion)

    # print("Total accuracy", accuracy)

    # plot_confusion(total_sequence_confusion)

    return losses


def run():
    directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-15-13-10-33-0-288/NC_003279.8"     # one-hot with anchors and reversal matrix Chr1 filtered 2820
    model_state_path = "output/training_2018-10-17-15-1-39-2-290/model_checkpoint_9"

    file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=directory, file_extension=".npz", sort=False)

    # Architecture parameters
    hidden_size = 256
    input_channels = 61      # 1-dimensional signal
    output_size = 1         # '-','A','C','T','G' one hot vector
    n_layers = 1

    # Hyperparameters
    learning_rate = 5e-4
    weight_decay = 1e-5
    dropout_rate = 0.1

    # Training parameters
    batch_size_train = 1
    n_batches = 10000

    scale_by_length = True
    checkpoint_interval = 200

    use_gpu = False

    data_loader = DataLoader(file_paths=file_paths, batch_size=batch_size_train, parse_batches=True, convert_to_distributions=True, convert_to_binary=True, use_gpu=use_gpu)
    print(data_loader.y_dtype)

    model = Decoder(hidden_size=hidden_size, input_size=input_channels, output_size=output_size, n_layers=n_layers, dropout_rate=dropout_rate, use_sigmoid=True)
    model.load_state_dict(torch.load(model_state_path))

    results_handler = ResultsHandler()

    # for alignment/pileup operations + conversions
    consensus_caller = ConsensusCaller(sequence_to_index=sequence_to_index, sequence_to_float=sequence_to_float)

    if use_gpu:
        model = model.cuda()

    test(model=model,
         data_loader=data_loader,
         n_batches=n_batches,
         results_handler=results_handler,
         checkpoint_interval=checkpoint_interval,
         consensus_caller=consensus_caller,
         use_gpu=use_gpu)


if __name__ == "__main__":
    run()

