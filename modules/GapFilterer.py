import sys
from os import path
sys.path.append(path.dirname(sys.path[0]))
from modules.train_test_utils import flatten_one_hot_tensor, plot_runlength_prediction
from handlers.FileManager import FileManager
from handlers.DataLoaderRunlength import DataLoader
from models.SimpleRnn import Decoder
import torch
import numpy
import gc


class GapFilterer:
    def __init__(self, model_state_path=None, threshold=0.005, use_gpu=True):
        if model_state_path is None:
            self.model_state_path = "/home/ryan/code/nanopore_assembly/models/parameters/filter_RNN_model_state_trained_chr5_celegans_1800_windows"
        else:
            self.model_state_path = model_state_path

        self.threshold = threshold

        # Architecture parameters
        self.hidden_size = 256
        self.input_channels = 61    # coverage, base frequencies (split by read direction), and length histogram (up to 50)
        self.output_size = 1        # binary classifier
        self.n_layers = 1
        self.dropout_rate = 0.1

        self.model = Decoder(hidden_size=self.hidden_size,
                             input_size=self.input_channels,
                             output_size=self.output_size,
                             n_layers=self.n_layers,
                             dropout_rate=self.dropout_rate,
                             use_sigmoid=True)

        self.model.load_state_dict(torch.load(self.model_state_path))

        self.use_gpu = use_gpu

        self.model.eval()

        gc.collect()

    def generate_filter_mask(self, x_pileup, y_pileup, x_repeat, reversal):
        if self.use_gpu:
            self.model.cuda()

        # (n,h,w) shape
        n_channels, height, width = x_pileup.shape

        _, x_pileup_distribution, x_repeat_distribution = \
            DataLoader.convert_pileup_to_distributions(x_pileup, x_repeat, reversal)

        x_pileup_distribution = numpy.expand_dims(x_pileup_distribution, axis=0)
        x_repeat_distribution = numpy.expand_dims(x_repeat_distribution, axis=0)

        # print(x_pileup_distribution.shape)
        # print(x_repeat_distribution.shape)

        if self.use_gpu:
            x_pileup_distribution = torch.cuda.FloatTensor(x_pileup_distribution)
            x_repeat_distribution = torch.cuda.FloatTensor(x_repeat_distribution)

        else:
            x_pileup_distribution = torch.FloatTensor(x_pileup_distribution)
            x_repeat_distribution = torch.FloatTensor(x_repeat_distribution)

        # print()
        # print("X PILEUP",x_pileup.shape)
        # print("Y PILEUP",y_pileup.shape)
        # print("X REPEAT",x_repeat.shape)
        # print("REVERSAL",reversal.shape)

        x = torch.cat([x_pileup_distribution, x_repeat_distribution], dim=2).reshape([1, 61, width])

        y_pileup_predict = self.model.forward(x)

        if self.use_gpu:
            y_pileup_predict = y_pileup_predict.cpu()

        gap_filter_mask = (y_pileup_predict.detach().numpy().reshape([width]) > self.threshold)

        return gap_filter_mask

    def filter_batch(self, batch, plot=False):
        """
        Use filter RNN to remove gap columns. Currently assumes batch size is always 1...
        :param batch:
        :return:
        """
        paths, x_pileup, y_pileup, x_repeat, y_repeat, reversal = batch

        batch_size, n_channels, height, width = x_pileup.shape

        x_pileup_n = x_pileup[0,:,:,:]
        y_pileup_n = y_pileup[0,:,:,:]
        x_repeat_n = x_repeat[0,:,:,:]
        y_repeat_n = y_repeat[0,:,:,:]
        reversal_n = reversal[0,:,:]

        mask = self.generate_filter_mask(x_pileup_n, y_pileup_n, x_repeat_n, reversal_n)

        x_pileup = x_pileup[:,:,:,mask]
        y_pileup = y_pileup[:,:,:,mask]
        x_repeat = x_repeat[:,:,:,mask]
        y_repeat = y_repeat[:,:,:,mask]
        reversal = reversal[:,:,mask]

        if plot:
            self.plot_pileups(x_pileup_n, y_pileup_n, x_repeat_n, y_repeat_n, mask)

        return x_pileup, y_pileup, x_repeat, y_repeat, reversal

    def plot_pileups(self, x_pileup_n, y_pileup_n, x_repeat_n, y_repeat_n, mask):
        n_channels, height, width = x_pileup_n.shape

        x_pileup_n_flat = flatten_one_hot_tensor(x_pileup_n)
        y_pileup_n_flat = flatten_one_hot_tensor(y_pileup_n)

        x_pileup_n = x_pileup_n_flat.reshape([height, width])
        y_pileup_n = y_pileup_n_flat.reshape([1, width])
        x_repeat_n = x_repeat_n.reshape([height, width])
        y_repeat_n = y_repeat_n.reshape([1, width])

        unfiltered_axes, height_ratios = plot_runlength_prediction(x_pileup=x_pileup_n,
                                                                   y_pileup=y_pileup_n,
                                                                   x_repeat=x_repeat_n,
                                                                   y_repeat=y_repeat_n)

        x_pileup_n_flat = x_pileup_n_flat[:,mask]
        y_pileup_n_flat = y_pileup_n_flat[:,mask]
        x_repeat_n = x_repeat_n[:,mask]
        y_repeat_n = y_repeat_n[:,mask]

        filtered_axes, height_ratios = plot_runlength_prediction(x_pileup=x_pileup_n_flat,
                                                                 y_pileup=y_pileup_n_flat,
                                                                 x_repeat=x_repeat_n,
                                                                 y_repeat=y_repeat_n)

        # fig, axes = pyplot.subplots(nrows=4, ncols=2, gridspec_kw={'height_ratios': height_ratios})
        #
        # unfiltered_axes[0].show()
        # pyplot.show()
        #
        # axes[0][0] = unfiltered_axes[0]
        # axes[1][0] = unfiltered_axes[1]
        # axes[2][0] = unfiltered_axes[2]
        # axes[3][0] = unfiltered_axes[3]
        #
        # axes[0][1] = filtered_axes[0]
        # axes[1][1] = filtered_axes[1]
        # axes[2][1] = filtered_axes[2]
        # axes[3][1] = filtered_axes[3]
        #
        # pyplot.show()
        # pyplot.close()


def test_filter(gap_filterer, data_loader, n_batches):
    for b, batch in enumerate(data_loader):
        # sys.stdout.write("\r %.2f%% COMPLETED  " % (100*b/n_batches))

        gap_filterer.filter_batch(batch, plot=True)

        if b == n_batches:
            break

    return


def run():
    directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-15-13-10-33-0-288/NC_003279.8"     # one-hot with anchors and reversal matrix Chr1 filtered 2820
    model_state_path = "output/training_2018-10-17-15-1-39-2-290/model_checkpoint_9"

    file_paths = FileManager.get_all_file_paths_by_type(parent_directory_path=directory, file_extension=".npz", sort=False)

    # Training parameters
    batch_size_train = 1
    n_batches = 1000

    threshold = 0.005

    data_loader = DataLoader(file_paths=file_paths,
                             batch_size=batch_size_train,
                             parse_batches=False,
                             convert_to_distributions=False,
                             use_gpu=False)

    gap_filterer = GapFilterer(model_state_path=model_state_path, threshold=threshold)

    test_filter(gap_filterer=gap_filterer,
                data_loader=data_loader,
                n_batches=n_batches)


if __name__ == "__main__":
    run()

