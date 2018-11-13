from modules.train_test_utils import *
from modules.pileup_utils import *
from handlers.FileManager import FileManager
from handlers.DataLoaderRunlength import DataLoader
from models.SimpleRnn import Decoder
from matplotlib import pyplot
import torch
from torch import optim
from torch import nn


def sequential_loss_BCE(y_predict, y, loss_fn):
    # x shape = (n, 5, length)
    # y shape = (n, 5, length)

    # print(y_predict.shape, y.shape)

    n, c, l = y_predict.shape

    y_predict = y_predict.reshape([l])
    # y_predict = softmax(y_predict)
    y_target = y.type(torch.FloatTensor).reshape([l])

    # print(y_target)
    # print(y_predict)

    loss = None

    for i in range(l):
        # print(y_predict[:,:,i].shape, y_target[:,:,i].shape)

        # print(y_predict[i], y_target[i])

        if i == 0:
            loss = loss_fn(y_predict[i], y_target[i])
        else:
            loss += loss_fn(y_predict[i], y_target[i])

    return loss


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


def train_batch(model, optimizer, x, y_pileup, y_repeat, loss_fn_repeat, loss_fn_base, scale_by_length):
    # Run forward calculation
    y_predict = model.forward(x)

    n, c, w = y_predict.shape

    # print(y_predict.shape)

    # y_repeat_predict = y_predict[:,5:,:]
    # y_pileup_predict = y_predict[:,:5,:]

    # print(y_repeat_predict.shape)
    # print(y_pileup_predict.shape)

    # print("loss")
    # print(y_repeat_predict.shape)
    # print(y_repeat.shape)

    # Compute loss.
    # repeat_loss = sequential_loss_MSE(y_repeat_predict, y_repeat.type(torch.FloatTensor), loss_fn_repeat)
    base_loss = sequential_loss_BCE(y_predict, y_pileup, loss_fn_base)

    # repeat_loss = sequential_loss_MSE(y_repeat_predict, y_repeat.type(torch.FloatTensor), loss_fn_repeat, scale_by_length)
    loss = base_loss

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

    return loss.item(), base_loss.item(), y_predict


def train(model, data_loader, optimizer, input_channels, n_batches, results_handler, checkpoint_interval, loss_fn_repeat, loss_fn_base, consensus_caller, scale_by_length, use_gpu=False):
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

        # print(x.shape)

        loss, base_loss, y_pileup_predict = train_batch(model=model,
                                                        optimizer=optimizer,
                                                        x=x,
                                                        y_pileup=y_pileup,
                                                        y_repeat=y_repeat,
                                                        loss_fn_repeat=loss_fn_repeat,
                                                        loss_fn_base=loss_fn_base,
                                                        scale_by_length=scale_by_length)

        losses.append(loss/width)
        print(b, loss/width)

        y_pileup_n = y_pileup[0,:,0,:]                  # ignore the depth dimension because y has always 1 "coverage"
        y_pileup_predict_n = y_pileup_predict[0,:,:]

        # print(paths[0])

        sequence_confusion, mismatches = sequential_binary_confusion(y_predict=y_pileup_predict_n, y=y_pileup_n)

        if total_sequence_confusion is None:
            total_sequence_confusion = sequence_confusion
        else:
            total_sequence_confusion += sequence_confusion

        if b % checkpoint_interval == 0:
            results_handler.save_model(model)

            accuracy = calculate_accuracy_from_confusion(total_sequence_confusion)
            print("Total accuracy", accuracy)

            pyplot.plot(losses)
            pyplot.show()
            pyplot.close()

            plot_binary_confusion(total_sequence_confusion)

            total_sequence_confusion = None

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
    # directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-15-13-10-33-0-288/NC_003283.11"     # one-hot with anchors and reversal matrix Chr5 filtered 2820
    directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-11-12-14-8-24-0-316/gi"     # one-hot with anchors and reversal matrix E. Coli

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
    n_batches = None

    scale_by_length = True
    checkpoint_interval = 200

    use_gpu = False

    data_loader = DataLoader(file_paths=file_paths, batch_size=batch_size_train, parse_batches=True, convert_to_distributions=True, convert_to_binary=True, use_gpu=use_gpu)
    print(data_loader.y_dtype)

    model = Decoder(hidden_size=hidden_size, input_size=input_channels, output_size=output_size, n_layers=n_layers, dropout_rate=dropout_rate, use_sigmoid=True)
    results_handler = ResultsHandler()

    # Initialize the optimizer with above parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Define the loss function
    loss_fn_repeat = nn.MSELoss()
    loss_fn_base = nn.BCELoss()

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
                   use_gpu=use_gpu)

    results_handler.save_model(model)
    results_handler.save_plot(losses)


if __name__ == "__main__":
    run()

