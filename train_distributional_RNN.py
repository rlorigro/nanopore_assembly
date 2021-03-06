from modules.train_test_utils import *
from modules.GapFilterer import GapFilterer
from modules.pileup_utils import *
from handlers.FileManager import FileManager
from handlers.DataLoaderRunlength import DataLoader
from models.SimpleRnn import Decoder
from matplotlib import pyplot
import torch
from torch import optim
from torch import nn


def sequential_loss_CE(y_predict, y, loss_fn):
    # x shape = (n, 5, length)
    # y shape = (n, 5, length)

    # print(y_predict.shape, y.shape)

    n, c, l = y_predict.shape

    y_target = torch.argmax(y, dim=1)

    # print(y_target.shape)
    # print(y_predict.shape)

    loss = None

    for i in range(l):
        # print(y_predict[:,:,i].shape, y_target[:,:,i].shape)

        if i == 0:
            loss = loss_fn(y_predict[:,:,i], y_target[0,:,i])
        else:
            loss += loss_fn(y_predict[:,:,i], y_target[0,:,i])

    return loss


def sequential_loss_MSE(y_predict, y, loss_fn, scale_by_runlength=False):
    # x shape = (n, 5, length)
    # y shape = (n, 5, length)

    n, c, l = y_predict.shape

    y_predict = y_predict.reshape([n,l])
    y = y.reshape([n,l])

    loss = None
    scale_factor = 1

    for i in range(l):
        predict = y_predict[:,i]
        target = y[:,i]

        if scale_by_runlength:
            scale_factor = max(1,int(target))   # scale loss by the true number of repeats at this position, with min=1
            # scale_factor = int(target)
            # print(scale_factor)

        if i == 0:
            loss = loss_fn(predict, target)*scale_factor
        else:
            loss += loss_fn(predict, target)*scale_factor

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
    # y shape = (length,)

    length = y_predict.shape[0]

    confusion = list()

    for l in range(length):
        target_index = int(y[l])
        predict_index = int(y_predict[l])

        confusion.append((target_index, predict_index))

    return confusion


def train_batch(model, optimizer, x, y_pileup, y_repeat, loss_fn_repeat, loss_fn_base, scale_by_length):
    # Run forward calculation
    y_predict = model.forward(x)

    n, c, w = y_predict.shape

    # print(y_predict.shape)

    y_repeat_predict = y_predict[:,5:,:]
    y_pileup_predict = y_predict[:,:5,:]

    # print(y_repeat_predict.shape)
    # print(y_pileup_predict.shape)

    # print("loss")
    # print(y_repeat_predict.shape)
    # print(y_repeat.shape)

    # Compute loss.
    # repeat_loss = sequential_loss_MSE(y_repeat_predict, y_repeat.type(torch.FloatTensor), loss_fn_repeat)
    base_loss = sequential_loss_CE(y_pileup_predict, y_pileup, loss_fn_base)
    repeat_loss = sequential_loss_MSE(y_repeat_predict, y_repeat.type(torch.FloatTensor), loss_fn_repeat, scale_by_length)

    loss = base_loss + repeat_loss*3

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    return loss.item(), base_loss.item(), repeat_loss.item(), y_pileup_predict, y_repeat_predict


def train(model, data_loader, optimizer, input_channels, n_batches, results_handler, checkpoint_interval, loss_fn_repeat, loss_fn_base, consensus_caller, gap_filterer=None, scale_by_length=False, use_gpu=False):
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
    losses = list()

    for b, batch in enumerate(data_loader):
        # sys.stdout.write("\r %.2f%% COMPLETED  " % (100*b/len(data_loader)))

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

        x_pileup_n = x_pileup[0,:,:,:]
        y_pileup_n = y_pileup[0,:,:,:]
        x_repeat_n = x_repeat[0,:,:,:]
        y_repeat_n = y_repeat[0,:,:,:]
        reversal_n = reversal[0,:,:]

        # (n,h,w) shape
        batch_size, n_channels, height, width = x_pileup.shape

        _, x_pileup_distribution, x_repeat_distribution = \
            DataLoader.convert_pileup_to_distributions(x_pileup_n, x_repeat_n, reversal_n)

        x_pileup_distribution = numpy.expand_dims(x_pileup_distribution, axis=0)
        x_repeat_distribution = numpy.expand_dims(x_repeat_distribution, axis=0)

        x_pileup_distribution = torch.FloatTensor(x_pileup_distribution)
        x_repeat_distribution = torch.FloatTensor(x_repeat_distribution)
        y_pileup = torch.FloatTensor(y_pileup)
        y_repeat = torch.FloatTensor(y_repeat)
        y_pileup_unfiltered = torch.FloatTensor(y_pileup_unfiltered)
        y_repeat_unfiltered = torch.FloatTensor(y_repeat_unfiltered)

        x = torch.cat([x_pileup_distribution, x_repeat_distribution], dim=2).reshape([1, 61, width])

        # print(x.shape)

        loss, base_loss, repeat_loss, y_pileup_predict, y_repeat_predict = train_batch(model=model,
                                                                                       optimizer=optimizer,
                                                                                       x=x,
                                                                                       y_pileup=y_pileup,
                                                                                       y_repeat=y_repeat,
                                                                                       loss_fn_repeat=loss_fn_repeat,
                                                                                       loss_fn_base=loss_fn_base,
                                                                                       scale_by_length=scale_by_length)

        losses.append(loss/width)
        print(b, loss/width)

        y_pileup_n = y_pileup[0,:,0,:]                          # ignore the depth dimension because y has always 1 "coverage"
        y_pileup_unfiltered_n = y_pileup_unfiltered[0,:,0,:]    # ignore the depth dimension because y has always 1 "coverage"
        y_pileup_predict_n = y_pileup_predict[0,:,:]
        y_repeat_n = y_repeat[0,:,0,:]                          # ignore the depth dimension because y has always 1 "coverage"
        y_repeat_unfiltered_n = y_repeat_unfiltered[0,:,0,:]    # ignore the depth dimension because y has always 1 "coverage"
        y_repeat_predict_n = y_repeat_predict[0,:,:]

        # print(y_pileup_n.shape)
        # print(y_pileup_n)

        y_pileup_predict_n_flattened = torch.argmax(y_pileup_predict_n, dim=0).data.numpy()
        y_repeat_predict_n_flattened = y_repeat_predict_n[0, :].data.numpy()

        y_pileup_n_flattened = torch.argmax(y_pileup_unfiltered_n, dim=0).data.numpy()
        y_repeat_n_flattened = y_repeat_unfiltered_n[0, :].data.numpy()

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

        # plot_confusion(sequence_confusion)

        if b % checkpoint_interval == 0:
            results_handler.save_model(model)

            repeat_predict = numpy.round(y_repeat_predict.reshape([1, width]).data.numpy(), 0).astype(numpy.uint8)
            repeat_target = numpy.round(y_repeat.reshape([1, width]).data.numpy(), 0).astype(numpy.uint8)
            repeat_comparison = numpy.concatenate([repeat_predict, repeat_target], axis=0).T

            print(repeat_comparison)
            accuracy = calculate_accuracy_from_confusion(total_expanded_confusion)
            print("Total accuracy", accuracy)

            # total_sequence_confusion = normalize_confusion_matrix(total_sequence_confusion)
            # total_expanded_confusion = normalize_confusion_matrix(total_expanded_confusion)

            pyplot.plot(losses)
            pyplot.show()
            pyplot.close()

            plot_repeat_confusion(total_repeat_confusion)
            plot_confusion(total_sequence_confusion)
            plot_confusion(total_expanded_confusion)

            total_sequence_confusion = None
            total_expanded_confusion = None
            total_repeat_confusion = list()

            # y_pileup_predict = y_pileup_predict.reshape([1, y_pileup_predict.shape[0]])
            # y_repeat_predict = y_repeat_predict.reshape([1, y_repeat_predict.shape[0]])
            #
            # x_pileup_n_flat = flatten_one_hot_tensor(x_pileup_n)
            # y_pileup_n_flat = flatten_one_hot_tensor(y_pileup_n)
            # y_pileup_predict_flat = flatten_one_hot_tensor(y_pileup_predict)
            #
            # plot_runlength_prediction(x_pileup=x_pileup_n_flat, y_pileup=y_pileup_n_flat, x_repeat=x_repeat_n, y_repeat=y_repeat_predict)

        #     plot_repeat_confusion(total_repeat_confusion)

    print()

    # total_sequence_confusion = normalize_confusion_matrix(total_sequence_confusion)
    # total_expanded_confusion = normalize_confusion_matrix(total_expanded_confusion)

    # accuracy = calculate_accuracy_from_confusion(total_expanded_confusion)

    # print("Total accuracy", accuracy)

    # plot_confusion(total_sequence_confusion)
    # plot_confusion(total_expanded_confusion)
    # plot_repeat_confusion(total_repeat_confusion)

    return losses


def run():
    directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-15-13-10-33-0-288/NC_003279.8"  # one-hot with anchors and reversal matrix Chr1 filtered 2820

    file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=directory, file_extension=".npz", sort=False)

    # Architecture parameters
    hidden_size = 512
    input_channels = 61      # 1-dimensional signal
    output_size = 5+1         # '-','A','C','T','G' one hot vector
    n_layers = 2

    # Hyperparameters
    learning_rate = 5e-4
    weight_decay = 1e-5
    dropout_rate = 0.1

    # Training parameters
    batch_size_train = 1
    n_batches = None

    scale_by_length = False
    checkpoint_interval = 200

    use_gpu = False

    data_loader = DataLoader(file_paths=file_paths, batch_size=batch_size_train, parse_batches=False, convert_to_distributions=False, use_gpu=use_gpu)
    model = Decoder(hidden_size=hidden_size, input_size=input_channels, output_size=output_size, n_layers=n_layers, dropout_rate=dropout_rate)

    gap_filterer = GapFilterer()

    results_handler = ResultsHandler()

    # Initialize the optimizer with above parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Define the loss function
    loss_fn_repeat = nn.MSELoss()
    loss_fn_base = nn.CrossEntropyLoss()

    # for alignment/pileup operations + conversions
    consensus_caller = ConsensusCaller(sequence_to_index=sequence_to_index, sequence_to_float=sequence_to_float)

    if use_gpu:
        model = model.cuda()

    losses = train(model=model,
                   input_channels=input_channels,
                   data_loader=data_loader,
                   optimizer=optimizer,
                   loss_fn_repeat=loss_fn_repeat,
                   loss_fn_base=loss_fn_base,
                   n_batches=n_batches,
                   results_handler=results_handler,
                   checkpoint_interval=checkpoint_interval,
                   consensus_caller=consensus_caller,
                   scale_by_length=scale_by_length,
                   gap_filterer=gap_filterer,
                   use_gpu=use_gpu)

    results_handler.save_model(model)
    results_handler.save_plot(losses)


if __name__ == "__main__":
    run()

