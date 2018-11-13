import sys
from os import walk, path
sys.path.append(path.dirname(sys.path[0]))
from modules.pileup_utils import trim_empty_rows, sequence_to_float, sequence_to_index, flatten_one_hot_tensor
from modules.train_test_utils import plot_runlength_prediction
from modules.ConsensusCaller import ConsensusCaller
from matplotlib import pyplot
import numpy
import torch


def plot_collapsed_encodings(pileup_matrix, pileup_repeat_matrix, reference_matrix, reference_repeat_matrix):
    print(pileup_matrix.shape, pileup_repeat_matrix.shape, reference_matrix.shape, reference_repeat_matrix.shape)

    ratio = pileup_matrix.shape[0] / reference_matrix.shape[0]
    fig, axes = pyplot.subplots(nrows=4, gridspec_kw={'height_ratios': [1, ratio, 1, ratio]})

    pileup_matrix = pileup_matrix.squeeze()
    pileup_repeat_matrix = pileup_repeat_matrix.squeeze()
    # reference_matrix = reference_matrix.squeeze()
    # reference_repeat_matrix = reference_repeat_matrix.squeeze()

    axes[3].imshow(pileup_repeat_matrix)
    axes[2].imshow(reference_repeat_matrix)
    axes[1].imshow(pileup_matrix)
    axes[0].imshow(reference_matrix)

    axes[3].set_ylabel("repeats")
    axes[2].set_ylabel("ref repeats")
    axes[1].set_ylabel("nucleotide")
    axes[0].set_ylabel("reference")

    pyplot.show()
    pyplot.close()


def get_all_file_paths_by_type(parent_directory_path, file_extension, sort=True):
    """
    Given a parent directory, iterate all files within, and return those that end in the extension provided by user.
    File paths returned in sorted order by default.
    :param parent_directory_path:
    :param file_extension:
    :param sort:
    :return:
    """
    all_files = list()

    for root, dirs, files in walk(parent_directory_path):
        sub_files = [path.join(root,subfile) for subfile in files if subfile.endswith(file_extension)]
        all_files.extend(sub_files)

    if sort:
        all_files.sort()

    return all_files


class DataLoader:
    def __init__(self, file_paths, batch_size, parse_batches=True, use_gpu=False, convert_to_frequency=False,
                 convert_repeats_to_counts=False, convert_to_distributions=False, convert_to_binary=False):
        self.file_paths = file_paths

        self.path_iterator = iter(file_paths)
        self.n_files = len(file_paths)
        self.files_loaded = 0

        self.use_gpu = use_gpu

        self.batch_size = batch_size
        self.parse_batches = parse_batches
        self.convert_to_frequency = convert_to_frequency
        self.convert_repeats_to_counts = convert_repeats_to_counts
        self.convert_to_distributions = convert_to_distributions
        self.convert_to_binary = convert_to_binary

        if self.use_gpu:
            print("USING GPU :)")
            self.x_dtype = torch.cuda.FloatTensor
            # y_dtype = torch.cuda.FloatTensor  # for MSE Loss or BCE loss
            self.y_dtype = torch.cuda.LongTensor      # for CE Loss

        else:
            self.x_dtype = torch.FloatTensor
            # y_dtype = torch.FloatTensor  # for MSE Loss or BCE loss
            self.y_dtype = torch.LongTensor      # for CE Loss

    def __len__(self):
        return self.n_files

    def load_next_file(self, path_cache, x_pileup_cache, y_pileup_cache, x_repeat_cache, y_repeat_cache, reversal_cache):
        """
        Assuming there is another file in the list of paths, load it and concatenate with the leftover entries from last
        :return:
        """
        next_path = next(self.path_iterator)

        # print(next_path)

        x_pileup = numpy.load(next_path)["x_pileup"]
        y_pileup = numpy.load(next_path)["y_pileup"]
        x_repeat = numpy.load(next_path)["x_repeat"]
        y_repeat = numpy.load(next_path)["y_repeat"]
        reversal = numpy.load(next_path)["reversal"]

        if self.convert_to_distributions:
            x_coverage, x_pileup, x_repeat = self.convert_pileup_to_distributions(x_pileup, x_repeat, reversal)

        if self.convert_to_binary:
            y_pileup = self.convert_pileup_target_to_binary(y_pileup)

        # add 3rd dimension
        x_pileup = numpy.expand_dims(x_pileup, axis=0)
        y_pileup = numpy.expand_dims(y_pileup, axis=0)
        x_repeat = numpy.expand_dims(x_repeat, axis=0)
        y_repeat = numpy.expand_dims(y_repeat, axis=0)
        reversal = numpy.expand_dims(reversal, axis=0).astype(numpy.float64)

        x_pileup_cache.append(x_pileup)
        y_pileup_cache.append(y_pileup)
        x_repeat_cache.append(x_repeat)
        y_repeat_cache.append(y_repeat)
        reversal_cache.append(reversal)

        path_cache.append(next_path)

        self.files_loaded += 1

        return path_cache, x_pileup_cache, y_pileup_cache, x_repeat_cache, y_repeat_cache, reversal_cache

    def load_batch(self):
        path_cache = list()
        x_pileup_cache = list()
        y_pileup_cache = list()
        x_repeat_cache = list()
        y_repeat_cache = list()
        reversal_cache = list()

        while len(x_pileup_cache) < self.batch_size and self.files_loaded < self.n_files:
            try:
                path_cache, x_pileup_cache, y_pileup_cache, x_repeat_cache, y_repeat_cache, reversal_cache = \
                    self.load_next_file(path_cache=path_cache,
                                        x_pileup_cache=x_pileup_cache,
                                        y_pileup_cache=y_pileup_cache,
                                        x_repeat_cache=x_repeat_cache,
                                        y_repeat_cache=y_repeat_cache,
                                        reversal_cache=reversal_cache)
            except ValueError as e:
                print(e)

        return path_cache, x_pileup_cache, y_pileup_cache, x_repeat_cache, y_repeat_cache, reversal_cache

    def parse_batch(self, x_pileup_batch, y_pileup_batch, x_repeat_batch, y_repeat_batch, reversal_batch):
        x_pileup_batch = torch.from_numpy(x_pileup_batch).type(self.x_dtype)    # + 1e-12  # offset to prevent 0 gradients
        y_pileup_batch = torch.from_numpy(y_pileup_batch).type(self.y_dtype)    # + 1e-12  # offset to prevent 0 gradients
        x_repeat_batch = torch.from_numpy(x_repeat_batch).type(self.x_dtype)    # + 1e-12  # offset to prevent 0 gradients
        y_repeat_batch = torch.from_numpy(y_repeat_batch).type(self.y_dtype)    # + 1e-12  # offset to prevent 0 gradients
        reversal_batch = torch.from_numpy(reversal_batch).type(self.x_dtype)    # + 1e-12  # offset to prevent 0 gradients

        return x_pileup_batch, y_pileup_batch, x_repeat_batch, y_repeat_batch, reversal_batch

    def convert_pileup_to_frequency(self, x_pileup_batch):
        caller = ConsensusCaller(sequence_to_float=sequence_to_float, sequence_to_index=sequence_to_index)

        batch_size, height, width = x_pileup_batch.shape
        # print("before", x_batch.shape)

        frequency_matrices = list()
        for b in range(batch_size):
            x_pileup = x_pileup_batch[b,:,:]

            x_pileup = trim_empty_rows(x_pileup, background_value=sequence_to_float["-"])

            normalized_frequencies = caller.get_normalized_frequencies(x_pileup)
            normalized_frequencies = numpy.expand_dims(normalized_frequencies, axis=0)

            frequency_matrices.append(normalized_frequencies)

        x_batch = numpy.concatenate(frequency_matrices, axis=0)

        return x_batch

    def convert_pileup_target_to_binary(self, y_pileup):
        y_pileup = y_pileup[:1,:,:]

        y_pileup = 1 - y_pileup

        return y_pileup

    @staticmethod
    def convert_pileup_to_distributions(x_pileup, x_repeat, reversal):
        if type(reversal) != numpy.bool:
            reversal = reversal.astype(numpy.bool)

        # print(x_pileup.shape)
        # print(x_repeat.shape)
        # print(reversal.shape)

        n_channels, height, width = x_pileup.shape

        # ---- character sums (split by strand direction) ----

        reverse = numpy.repeat(reversal[numpy.newaxis,:,:], n_channels, axis=0)
        forward = numpy.invert(reverse)

        forward_characters = x_pileup*forward
        reverse_characters = x_pileup*reverse

        forward_sums = numpy.sum(forward_characters, axis=1)
        reverse_sums = numpy.sum(reverse_characters, axis=1)[1:,:]

        x_pileup_distribution = numpy.concatenate([forward_sums, reverse_sums], axis=0)

        x_coverage = numpy.expand_dims(numpy.sum(x_pileup_distribution, axis=0), axis=0)

        # ---- repeat distribution from 0-50 -----------------

        x_repeat = x_repeat.reshape([height, width]).astype(numpy.int64)

        # vectorized bincount via stackoverflow:
        # https://stackoverflow.com/questions/46256279/bin-elements-per-row-vectorized-2d-bincount-for-numpy
        N = 50 + 1
        a_offs = x_repeat.T + numpy.arange(x_repeat.T.shape[0])[:, None]*N
        x_repeat_distribution = numpy.bincount(a_offs.ravel(), minlength=x_repeat.T.shape[0]*N).reshape(-1, N).T

        x_pileup_distribution = x_pileup_distribution.astype(numpy.float64) / x_coverage
        x_repeat_distribution = x_repeat_distribution.astype(numpy.float64) / x_coverage
        x_coverage = x_coverage.astype(numpy.float64)/1000

        x_pileup_distribution = numpy.concatenate([x_coverage, x_pileup_distribution], axis=0)

        x_pileup_distribution = numpy.expand_dims(x_pileup_distribution, axis=0)
        x_repeat_distribution = numpy.expand_dims(x_repeat_distribution, axis=0)

        # ratio = x_repeat_distribution.shape[0] / x_pileup_distribution.shape[0]
        # ratio2 = x_coverage.shape[0] / x_pileup_distribution.shape[0]
        # fig, axes = pyplot.subplots(nrows=3, gridspec_kw={'height_ratios': [ratio2, ratio, 1]})
        # axes[2].imshow(x_pileup_distribution)
        # axes[1].imshow(x_repeat_distribution)
        # axes[0].imshow(x_coverage)
        # pyplot.show()
        # pyplot.close()

        return x_coverage, x_pileup_distribution, x_repeat_distribution

    def convert_repeat_matrix_to_counts(self, x_pileup_batch, x_repeat_batch):
        caller = ConsensusCaller(sequence_to_float=sequence_to_float, sequence_to_index=sequence_to_index)

        batch_size, height, width = x_repeat_batch.shape
        # print("before", x_batch.shape)

        repeat_matrices = list()
        for b in range(batch_size):
            x_repeat = x_repeat_batch[b,:,:]
            x_pileup = x_pileup_batch[b,:,:]

            x_pileup = trim_empty_rows(x_pileup, background_value=sequence_to_float["-"])
            x_repeat = trim_empty_rows(x_repeat, background_value=sequence_to_float["-"])

            repeat_counts = caller.get_avg_repeat_counts(pileup_matrix=x_pileup, repeat_matrix=x_repeat)
            repeat_counts = numpy.expand_dims(repeat_counts, axis=0)

            repeat_matrices.append(repeat_counts)

        x_batch = numpy.concatenate(repeat_matrices, axis=0)

        return x_batch

    def __next__(self):
        """
        Get the next batch data. DOES NOT RETURN FINAL BATCH IF != BATCH SIZE
        :return:
        """
        path_cache, x_pileup_cache, y_pileup_cache, x_repeat_cache, y_repeat_cache, reversal_cache = self.load_batch()

        if len(path_cache) > 0:
            x_pileup_batch = numpy.concatenate(x_pileup_cache, axis=0)
            y_pileup_batch = numpy.concatenate(y_pileup_cache, axis=0)
            x_repeat_batch = numpy.concatenate(x_repeat_cache, axis=0)
            y_repeat_batch = numpy.concatenate(y_repeat_cache, axis=0)
            reversal_batch = numpy.concatenate(reversal_cache, axis=0)

            assert x_pileup_batch.shape[0] == self.batch_size
            assert y_pileup_batch.shape[0] == self.batch_size
            assert x_repeat_batch.shape[0] == self.batch_size
            assert y_repeat_batch.shape[0] == self.batch_size

            if self.convert_repeats_to_counts:
                x_repeat_batch = self.convert_repeat_matrix_to_counts(x_pileup_batch=x_pileup_batch, x_repeat_batch=x_repeat_batch)

            if self.convert_to_frequency:
                x_pileup_batch = self.convert_pileup_to_frequency(x_pileup_batch)

            if self.parse_batches:
                x_pileup_batch, y_pileup_batch, x_repeat_batch, y_repeat_batch, reversal_batch = \
                    self.parse_batch(x_pileup_batch, y_pileup_batch, x_repeat_batch, y_repeat_batch, reversal_batch)

        else:
            # reset dataloader iterator
            self.__init__(file_paths=self.file_paths,
                          batch_size=self.batch_size,
                          parse_batches=self.parse_batches,
                          use_gpu=self.use_gpu,
                          convert_to_frequency=self.convert_to_frequency)

            # end epoch
            raise StopIteration

        return path_cache, x_pileup_batch, y_pileup_batch, x_repeat_batch, y_repeat_batch, reversal_batch

    def __iter__(self):
        return self


if __name__ == "__main__":
    directory = "/home/ryan/code/nanopore_assembly/output/spoa_pileup_generation_2018-10-15-13-10-33-0-288/NC_003279.8"
    file_paths = get_all_file_paths_by_type(parent_directory_path=directory, file_extension=".npz")

    data_loader = DataLoader(file_paths=file_paths, batch_size=1, parse_batches=False, convert_to_distributions=True, convert_to_binary=True)

    for path_cache, x_pileup, y_pileup, x_repeat, y_repeat, reversal in data_loader:
        print(x_pileup.shape, y_pileup.shape, x_repeat.shape, y_repeat.shape)

        plot_collapsed_encodings(pileup_matrix=flatten_one_hot_tensor(x_pileup[0,:,:]),
                                 reference_matrix=flatten_one_hot_tensor(y_pileup[0,:,:]),
                                 pileup_repeat_matrix=flatten_one_hot_tensor(x_repeat[0,:,:]),
                                 reference_repeat_matrix=flatten_one_hot_tensor(y_repeat[0,:,:]))
