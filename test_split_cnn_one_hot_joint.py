from modules.train_test_utils import *
from modules.pileup_utils import *
from handlers.FileManager import FileManager
from handlers.DataLoaderRunlength import DataLoader
from models.SplitCnn import EncoderDecoder
from matplotlib import pyplot
import torch
from torch import optim
from torch import nn


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
    # y shape = (length,)

    length = y_predict.shape[0]

    confusion = list()

    for l in range(length):
        target_index = int(y[l])
        predict_index = int(y_predict[l])

        confusion.append((target_index, predict_index))

    return confusion


# def train_batch(model, optimizer, x, y_pileup, y_repeat, loss_fn_repeat, loss_fn_base, scale_by_length):
#     # Run forward calculation
#
#     # print(y_repeat_predict.shape)
#     # print(y_pileup_predict.shape)
#
#     # print("loss")
#     # print(y_repeat_predict.shape)
#     # print(y_repeat.shape)
#
#     # Compute loss.
#     # repeat_loss = sequential_loss_MSE(y_repeat_predict, y_repeat.type(torch.FloatTensor), loss_fn_repeat)
#     base_loss = sequential_loss_CE(y_pileup_predict, y_pileup, loss_fn_base)
#     repeat_loss = sequential_loss_MSE(y_repeat_predict, y_repeat.type(torch.FloatTensor), loss_fn_repeat, scale_by_length)
#
#     loss = base_loss + repeat_loss
#
#     # Before the backward pass, use the optimizer object to zero all of the
#     # gradients for the variables it will update (which are the learnable weights
#     # of the model)
#     optimizer.zero_grad()
#
#     # Backward pass: compute gradient of the loss with respect to model
#     # parameters
#     loss.backward()
#
#     # Calling the step function on an Optimizer makes an update to its
#     # parameters
#     optimizer.step()
#
#     return loss.item(), base_loss.item(), repeat_loss.item(), y_pileup_predict, y_repeat_predict


def test(model, data_loader, input_channels, n_batches, results_handler, checkpoint_interval, consensus_caller, scale_by_length, use_gpu=False):
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
    total_expanded_confusion = None
    total_repeat_confusion = list()

    for b, batch in enumerate(data_loader):
        # sys.stdout.write("\r %.2f%% COMPLETED  " % (100*b/len(data_loader)))
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

        x = torch.cat([x_pileup, x_repeat, reversal], dim=1)

        padding_height = MAX_COVERAGE - height

        if padding_height > 0:
            padding = torch.zeros([batch_size, input_channels, padding_height, width])
            x = torch.cat([x, padding], dim=2)

        y_predict = model.forward(x)

        n, c, w = y_predict.shape

        # print(y_predict.shape)

        y_repeat_predict = y_predict[:, 5:, :]
        y_pileup_predict = y_predict[:, :5, :]

        y_pileup_n = y_pileup[0,:,0,:]                  # ignore the depth dimension because y has always 1 "coverage"
        y_pileup_predict_n = y_pileup_predict[0,:,:]
        y_repeat_n = y_repeat[0,:,0,:]                  # ignore the depth dimension because y has always 1 "coverage"
        y_repeat_predict_n = y_repeat_predict[0,:,:]

        # print(y_pileup_n.shape)
        # print(y_pileup_n)

        y_pileup_predict_n_flattened = torch.argmax(y_pileup_predict_n, dim=0).data.numpy()
        y_repeat_predict_n_flattened = y_repeat_predict_n[0, :].data.numpy()
        y_pileup_n_flattened = torch.argmax(y_pileup_n, dim=0).data.numpy()
        y_repeat_n_flattened = y_repeat_n[0, :].data.numpy()

        # print(y_pileup_n_flattened)
        # print(y_pileup_predict_n.shape)

        print(paths[0])

        # decode as string to compare with non-runlength version
        expanded_consensus_string = \
            consensus_caller.expand_collapsed_consensus_as_string(consensus_indices=y_pileup_predict_n_flattened,
                                                                  repeat_consensus_integers=y_repeat_predict_n_flattened,
                                                                  ignore_spaces=True)

        # decode as string to compare with non-runlength version
        expanded_reference_string = \
            consensus_caller.expand_collapsed_consensus_as_string(consensus_indices=y_pileup_n_flattened,
                                                                  repeat_consensus_integers=y_repeat_n_flattened,
                                                                  ignore_spaces=True)

        # print(expanded_reference_string)
        # print(expanded_consensus_string)

        if len(expanded_consensus_string) == 0:
            expanded_consensus_string = '-'

        # realign strings to each other and convert to one hot
        y_pileup_predict_expanded, y_pileup_expanded = \
            realign_consensus_to_reference(consensus_sequence=expanded_consensus_string,
                                           ref_sequence=expanded_reference_string,
                                           print_alignment=True)

        y_pileup_predict_expanded = y_dtype(y_pileup_predict_expanded)
        y_pileup_expanded = y_dtype(y_pileup_expanded)

        expanded_confusion, _ = sequential_confusion(y_predict=y_pileup_predict_expanded, y=y_pileup_expanded)

        # y_pileup_predict = torch.FloatTensor(y_pileup_predict)
        # y_pileup_n = torch.FloatTensor(y_pileup_n)

        sequence_confusion, mismatches = sequential_confusion(y_predict=y_pileup_predict_n, y=y_pileup_n)

        repeat_predict = numpy.round(y_repeat_predict.reshape([width]).data.numpy(), 0).astype(numpy.uint8)
        repeat_target = numpy.round(y_repeat.reshape([width]).data.numpy(), 0).astype(numpy.uint8)

        repeat_confusion = sequential_repeat_confusion(y_predict=repeat_predict, y=repeat_target)

        total_repeat_confusion.extend(repeat_confusion)

        if total_sequence_confusion is None:
            total_sequence_confusion = sequence_confusion
        else:
            total_sequence_confusion += sequence_confusion

        if total_expanded_confusion is None:
            total_expanded_confusion = expanded_confusion
        else:
            total_expanded_confusion += expanded_confusion

        # if b % checkpoint_interval == 0:
        #     results_handler.save_model(model)
        #
        #     repeat_predict = numpy.round(y_repeat_predict.reshape([1, width]).data.numpy(), 0).astype(numpy.uint8)
        #     repeat_target = numpy.round(y_repeat.reshape([1, width]).data.numpy(), 0).astype(numpy.uint8)
        #     repeat_comparison = numpy.concatenate([repeat_predict, repeat_target], axis=0).T
        #
        #     print(repeat_comparison)
        #     accuracy = calculate_accuracy_from_confusion(total_expanded_confusion)
        #     print("Total accuracy", accuracy)
        #
        #     total_sequence_confusion = normalize_confusion_matrix(total_sequence_confusion)
        #     total_expanded_confusion = normalize_confusion_matrix(total_expanded_confusion)
        #
        #     plot_repeat_confusion(total_repeat_confusion)
        #     plot_confusion(total_sequence_confusion)
        #     plot_confusion(total_expanded_confusion)
        #
        #     total_sequence_confusion = None
        #     total_expanded_confusion = None
        #     total_repeat_confusion = list()

            # y_pileup_predict = y_pileup_predict.reshape([1, y_pileup_predict.shape[0]])
            # y_repeat_predict = y_repeat_predict.reshape([1, y_repeat_predict.shape[0]])
            #
            # x_pileup_n_flat = flatten_one_hot_tensor(x_pileup_n)
            # y_pileup_n_flat = flatten_one_hot_tensor(y_pileup_n)
            # y_pileup_predict_flat = flatten_one_hot_tensor(y_pileup_predict)
            #
            # plot_runlength_prediction(x_pileup=x_pileup_n_flat, y_pileup=y_pileup_n_flat, x_repeat=x_repeat_n, y_repeat=y_repeat_predict)

        #     plot_repeat_confusion(total_repeat_confusion)
        #     total_repeat_confusion = list()

        if b > 500:
            break
    print()

    # total_sequence_confusion = normalize_confusion_matrix(total_sequence_confusion)
    # total_expanded_confusion = normalize_confusion_matrix(total_expanded_confusion)

    accuracy = calculate_accuracy_from_confusion(total_expanded_confusion)

    print("Total accuracy", accuracy)

    plot_confusion(total_sequence_confusion)
    plot_confusion(total_expanded_confusion)
    plot_repeat_confusion(total_repeat_confusion)


def run():
    # directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-2-10-43-22-1-275/NC_003282.8"  # Training Chr4
    directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-2-10-43-22-1-275/NC_003279.8"  # Testing Chr1

    model_state_path = "/home/ryan/code/nanopore_assembly/output/training_2018-10-9-17-51-4-1-282/model_checkpoint_10"

    file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=directory, file_extension=".npz", sort=False)

    # Architecture parameters
    hidden_size = 512
    input_channels = 7      # 1-dimensional signal
    output_size = 5+1         # '-','A','C','T','G' one hot vector
    n_layers = 1

    # Hyperparameters
    learning_rate = 1e-3
    weight_decay = 1e-5
    dropout_rate = 0.1

    # Training parameters
    batch_size_train = 1
    n_batches = None

    scale_by_length = False

    checkpoint_interval = 50

    use_gpu = False

    data_loader = DataLoader(file_paths=file_paths, batch_size=batch_size_train, parse_batches=True, use_gpu=use_gpu)
    model = EncoderDecoder(hidden_size=hidden_size, input_size=input_channels, output_size=output_size, n_layers=n_layers, dropout_rate=dropout_rate)
    results_handler = ResultsHandler()


    model.load_state_dict(torch.load(model_state_path))

    # for alignment/pileup operations + conversions
    consensus_caller = ConsensusCaller(sequence_to_index=sequence_to_index, sequence_to_float=sequence_to_float)

    if use_gpu:
        model = model.cuda()

    losses = test(model=model,
                   input_channels=input_channels,
                   data_loader=data_loader,
                   n_batches=n_batches,
                   results_handler=results_handler,
                   checkpoint_interval=checkpoint_interval,
                   consensus_caller=consensus_caller,
                   scale_by_length=scale_by_length,
                   use_gpu=use_gpu)

    results_handler.save_model(model)
    results_handler.save_plot(losses)



if __name__ == "__main__":
    run()
