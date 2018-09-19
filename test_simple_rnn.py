from handlers.DataLoader import DataLoader
from modules.pileup_utils import sequence_to_float, sequence_to_index
from modules.train_test_utils import *
from modules.ConsensusCaller import *
from models.SimpleRnn import Decoder
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

            x_n = trim_empty_rows(x_n, background_value=sequence_to_float["-"])

            y_predict_n = consensus_caller.call_consensus_as_one_hot(x_n)
            y_predict_n = torch.FloatTensor(y_predict_n)

            confusion = sequential_confusion(y_predict=y_predict_n, y=y_n)

            # normalized_frequencies = consensus_caller.call_consensus_as_normalized_frequencies(x_n)
            # plot_consensus_prediction(x=x_n,y=y_n,y_predict=normalized_frequencies)

            if total_confusion is None:
                total_confusion = confusion
            else:
                total_confusion += confusion

    total_confusion = normalize_confusion_matrix(total_confusion)
    plot_confusion(total_confusion)


def run():
    model_state_path = "/home/ryan/code/nanopore_assembly/output/simple_rnn_2_layer_32_hidden_celegans/model_checkpoint_10"
    # directory = "/home/ryan/code/nanopore_assembly/output/chr1_800k-1200k_standard_20width/chr1/test"     # no variants, chr1:1200k 200 window test
    directory = "/home/ryan/code/nanopore_assembly/output/celegans_250_window_test_1m"       # celegans 50 windows

    file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=directory, file_extension=".npz", sort=False)

    # Architecture parameters
    hidden_size = 32
    input_channels = 5      # 1-dimensional signal
    output_size = 5         # '-','A','C','T','G' one hot vector
    n_layers = 2

    # Hyperparameters
    dropout_rate = 0.1

    # Training parameters
    batch_size_train = 1
    n_batches = 8000

    checkpoint_interval = 1000

    data_loader = DataLoader(file_paths=file_paths, batch_size=batch_size_train, convert_to_frequency=True)

    model = Decoder(hidden_size=hidden_size, input_size=input_channels, output_size=output_size, n_layers=n_layers, dropout_rate=dropout_rate)
    model.load_state_dict(torch.load(model_state_path))

    test(model=model, data_loader=data_loader)

    consensus_caller = ConsensusCaller(sequence_to_float=sequence_to_float, sequence_to_index=sequence_to_index)

    test_consensus(consensus_caller=consensus_caller, data_loader=data_loader)


if __name__ == "__main__":
    run()
