from handlers.FileManager import FileManager
from handlers.DataLoader import DataLoader
from modules.train_test_utils import plot_prediction
from models.SplitCnn import EncoderDecoder
from matplotlib import pyplot
from torch import nn
from torch import optim
from os import path
import torch
import datetime


class ResultsHandler:
    def __init__(self):
        self.datetime_string = '-'.join(list(map(str, datetime.datetime.now().timetuple()))[:-1])
        self.subdirectory_name = "training_" + self.datetime_string

        self.output_directory_name = "output/"
        self.directory = path.join(self.output_directory_name, self.subdirectory_name)

        self.n_checkpoints = 0

        FileManager.ensure_directory_exists(self.directory)

    def save_plot(self, losses):
        loss_plot_filename = path.join(self.directory, "loss.png")

        figure = pyplot.figure()
        axes = pyplot.axes()
        axes.plot(losses)
        pyplot.savefig(loss_plot_filename)

    def save_model(self, model):
        self.n_checkpoints += 1

        model_filename = path.join(self.directory, "model_checkpoint_%d" % self.n_checkpoints)
        torch.save(model.state_dict(), model_filename)

    def save_config(self, model):
        pass


def sequential_loss_CE(y_predict, y, loss_fn):
    # x shape = (n, 5, length)
    # y shape = (n, 5, length)

    n, c, l = y_predict.shape

    # print(n,c,l)

    y_target = torch.argmax(y, dim=1)

    loss = None

    for i in range(l):
        if i == 0:
            # print(y[:,:,i])
            # print(torch.nn.functional.softmax(y_predict[:,:,i]))

            loss = loss_fn(y_predict[:,:,i], y_target[:,i])
        else:
            loss += loss_fn(y_predict[:,:,i], y_target[:,i])

    return loss


def sequential_loss_MSE(y_predict, y_target, loss_fn):
    # x shape = (n, 5, length)
    # y shape = (n, 5, length)

    n, c, l = y_predict.shape

    loss = None

    for i in range(l):
        if i == 0:
            # print(y_predict[:,:,i])
            # print(y_target[:,:,i])

            loss = loss_fn(y_predict[:,:,i], y_target[:,:,i])
        else:
            loss += loss_fn(y_predict[:,:,i], y_target[:,:,i])

    return loss


def train_batch(model, x, y, optimizer, loss_fn):
    # Run forward calculation
    y_predict = model.forward(x)

    # Compute loss.
    # loss = loss_fn(y_predict, y)
    loss = sequential_loss_CE(y_predict, y, loss_fn)

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

    return loss.item(), y_predict


def train(model, data_loader, optimizer, loss_fn, n_batches, results_handler, checkpoint_interval):
    model.train()

    losses = list()

    for b, batch in enumerate(data_loader):
        paths, x, y = batch

        # print("x1", x.shape)
        # print("y1", y.shape)

        n, h, w = x.shape
        x = x.view([n,1,h,w])

        # n, h, w = y.shape
        # y = y.view([n,1,h,w])

        # print("x2", x.shape)
        # print("y2", y.shape)

        # expected convolution input = (batch, channel, H, W)
        # y_predict shape = (batch_size, seq_len, hidden_size*num_directions)
        loss, y_predict = train_batch(model=model, x=x, y=y, optimizer=optimizer, loss_fn=loss_fn)
        avg_loss = loss/w
        losses.append(avg_loss)

        print(b, avg_loss)

        if loss > 100:
            print("Warning: extreme loss observed for training example:", paths[0])

        if b % checkpoint_interval == 0:
            results_handler.save_model(model)

            print(paths[0])
            plot_prediction(x=x,y=y,y_predict=y_predict)

            pyplot.plot(losses)
            pyplot.show()
            pyplot.close()

    return losses


def run(load_model=False, model_state_path=None):
    directory = "/home/ryan/code/nanopore_assembly/output/chr1_800k-1200k_standard_20width/chr1/train"     # spoa 2 pass arbitray region 2500 windows

    file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=directory, file_extension=".npz", sort=False)

    results_handler = ResultsHandler()

    # Architecture parameters
    hidden_size = 128
    input_channels = 1      # 1-dimensional signal
    output_size = 5         # '-','A','C','T','G' one hot vector
    n_layers = 1

    # Hyperparameters
    learning_rate = 1e-3
    weight_decay = 1e-5
    dropout_rate = 0.1

    # Training parameters
    batch_size_train = 1
    n_batches = None

    checkpoint_interval = 1000

    data_loader = DataLoader(file_paths=file_paths, batch_size=batch_size_train)
    model = EncoderDecoder(hidden_size=hidden_size, input_size=input_channels, output_size=output_size, n_layers=n_layers, dropout_rate=dropout_rate)

    # Initialize the optimizer with above parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Define the loss function
    # loss_fn = nn.MSELoss()
    loss_fn = nn.CrossEntropyLoss()

    if load_model:
        # get weight parameters from saved model state
        model.load_state_dict(torch.load(model_state_path))

    # Train and get the resulting loss per iteration
    losses = train(model=model,
                   data_loader=data_loader,
                   optimizer=optimizer,
                   loss_fn=loss_fn,
                   n_batches=n_batches,
                   results_handler=results_handler,
                   checkpoint_interval=checkpoint_interval)

    # test(model=model,
    #      data_loader=data_loader,
    #      n_batches=4)

    results_handler.save_model(model)
    results_handler.save_plot(losses)

    print(model)


# def test_saved_model(model_state_path):
#     data_loader = initialize_signal_generator()
#
#     # Define signal simulator parameters
#     sequence_nucleotide_length = 8
#
#     # Define architecture parameters
#     hidden_size = 3*sequence_nucleotide_length
#     input_size = 1      # 1-dimensional signal
#     n_layers = 3
#
#     model = Autoencoder(hidden_size=hidden_size, input_size=input_size, n_layers=n_layers, dropout_rate=0)
#     model.load_state_dict(torch.load(model_state_path))
#
#     print(model)
#
#     test(model=model,
#          data_loader=data_loader,
#          n_batches=4)
#
#     predict_encoding(model=model,
#                      data_loader=data_loader,
#                      n_batches=4)


if __name__ == "__main__":
    run()

    # model_path = "/home/ryan/code/nanopore_signal_simulation/output/2018-7-25-12-7-43-2-206/model_checkpoint_41"
    # test_saved_model(model_path)

    # run(load_model=True, model_state_path=model_path)

